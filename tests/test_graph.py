"""Graph tests: fake LLM clients, in-memory or SQLite checkpoints, no API key."""

from __future__ import annotations

import sqlite3

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from agent import Shop, build_graph
from agent.fake import DemoClient, ScriptedClient, reply, tool_call


def config(thread: str = "t1") -> dict:
    return {"configurable": {"thread_id": thread}, "recursion_limit": 20}


def user(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


def last_answer(graph, cfg) -> str:
    return graph.get_state(cfg).values["messages"][-1]["content"]


def test_read_only_tools_run_without_pausing():
    client = ScriptedClient([
        reply(tool_calls=[tool_call("c1", "lookup_order", {"order_id": "A1037"})]),
        reply("It has shipped."),
    ])
    graph = build_graph(client, "fake", Shop(), InMemorySaver())
    graph.invoke(user("Where is A1037?"), config())

    assert graph.get_state(config()).next == ()  # finished, not paused
    tool_msg = client.calls[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and "shipped" in tool_msg["content"]


def test_refund_pauses_and_does_not_run_before_approval():
    shop = Shop()
    client = ScriptedClient([
        reply(tool_calls=[tool_call("r1", "issue_refund", {"order_id": "A1042", "amount": 18500, "reason": "broken"})]),
    ])
    graph = build_graph(client, "fake", shop, InMemorySaver())
    result = graph.invoke(user("Refund A1042"), config())

    actions = result["__interrupt__"][0].value["actions"]
    assert actions == [{"id": "r1", "tool": "issue_refund", "args": {"order_id": "A1042", "amount": 18500, "reason": "broken"}}]
    assert graph.get_state(config()).next == ("review",)
    assert shop.refunds == []  # nothing happened yet


def test_approve_runs_the_action():
    shop = Shop()
    client = ScriptedClient([
        reply(tool_calls=[tool_call("r1", "issue_refund", {"order_id": "A1042", "amount": 18500, "reason": "broken"})]),
        reply("Refunded."),
    ])
    graph = build_graph(client, "fake", shop, InMemorySaver())
    graph.invoke(user("Refund A1042"), config())
    graph.invoke(Command(resume={"r1": {"action": "approve"}}), config())

    assert shop.refunds[0]["amount"] == 18500
    state = graph.get_state(config()).values
    assert state["audit"] == [{"tool": "issue_refund", "args": {"order_id": "A1042", "amount": 18500, "reason": "broken"},
                               "decision": "approved", "reason": ""}]
    assert last_answer(graph, config()) == "Refunded."


def test_reject_sends_reason_back_to_model():
    shop = Shop()
    client = ScriptedClient([
        reply(tool_calls=[tool_call("e1", "send_email", {"to": "amaka@example.com", "subject": "Hi", "body": "x"})]),
        reply("OK, I won't email her."),
    ])
    graph = build_graph(client, "fake", shop, InMemorySaver())
    graph.invoke(user("Email Amaka"), config())
    graph.invoke(Command(resume={"e1": {"action": "reject", "reason": "wrong tone"}}), config())

    assert shop.outbox == []
    tool_msg = client.calls[1]["messages"][-1]
    assert '"rejected"' in tool_msg["content"] and "wrong tone" in tool_msg["content"]


def test_edit_runs_with_reviewer_arguments():
    shop = Shop()
    client = ScriptedClient([
        reply(tool_calls=[tool_call("r1", "issue_refund", {"order_id": "A1042", "amount": 18500, "reason": "broken"})]),
        reply("Partial refund done."),
    ])
    graph = build_graph(client, "fake", shop, InMemorySaver())
    graph.invoke(user("Refund A1042"), config())
    edited = {"order_id": "A1042", "amount": 9250, "reason": "one mug broken"}
    graph.invoke(Command(resume={"r1": {"action": "edit", "args": edited}}), config())

    assert shop.refunds[0]["amount"] == 9250
    assert graph.get_state(config()).values["audit"][0]["decision"] == "edited"


def test_missing_decision_fails_closed():
    shop = Shop()
    client = ScriptedClient([
        reply(tool_calls=[tool_call("r1", "issue_refund", {"order_id": "A1042", "amount": 100, "reason": "x"})]),
        reply("Not done."),
    ])
    graph = build_graph(client, "fake", shop, InMemorySaver())
    graph.invoke(user("Refund A1042"), config())
    graph.invoke(Command(resume={}), config())  # reviewer answered nothing for r1

    assert shop.refunds == []


def test_pause_survives_a_restart(tmp_path):
    """Pause in one 'process', resume from a brand-new graph on the same SQLite file."""
    db = tmp_path / "state.sqlite"

    first = build_graph(DemoClient(), "demo", Shop(), SqliteSaver(sqlite3.connect(db, check_same_thread=False)))
    first.invoke(user("Refund order A1042, the mugs arrived broken"), config("shop-1"))
    assert first.get_state(config("shop-1")).next == ("review",)

    shop = Shop()
    second = build_graph(DemoClient(), "demo", shop, SqliteSaver(sqlite3.connect(db, check_same_thread=False)))
    pending = second.get_state(config("shop-1")).tasks[0].interrupts[0].value["actions"]
    decisions = {a["id"]: {"action": "approve"} for a in pending}
    second.invoke(Command(resume=decisions), config("shop-1"))

    assert shop.refunds[0]["order_id"] == "A1042"
    assert shop.outbox[0]["to"] == "amaka@example.com"
    assert "Refunded ₦18,500" in last_answer(second, config("shop-1"))


@pytest.mark.parametrize(("text", "expected"), [
    ("Where is order A1037?", "shipped, expected 2026-11-20"),
    ("Which orders does chioma@example.com have?", "A1029"),
    ("Refund order A1029", "already refunded in full"),
    ("hello", "I can look up orders"),
])
def test_demo_conversations_finish_without_approval(text, expected):
    graph = build_graph(DemoClient(), "demo", Shop(), InMemorySaver())
    graph.invoke(user(text), config())
    assert expected in last_answer(graph, config())


def test_demo_refund_rejected_moves_no_money():
    shop = Shop()
    graph = build_graph(DemoClient(), "demo", shop, InMemorySaver())
    result = graph.invoke(user("Refund A1042 please"), config())
    ids = [a["id"] for a in result["__interrupt__"][0].value["actions"]]
    graph.invoke(Command(resume={i: {"action": "reject", "reason": "needs photo"} for i in ids}), config())

    assert shop.refunds == [] and shop.outbox == []
    assert "not approved (needs photo)" in last_answer(graph, config())
