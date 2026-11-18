"""The LangGraph state machine.

    START → agent ──(no tool calls)──────────────────────────→ END
              │
              ├─(only read-only tools)──────────→ tools → agent
              │
              └─(a risky tool)→ review ‖ (interrupt) → tools → agent

`review` calls `interrupt()`, which saves the whole state to the
checkpointer and stops the run. The run continues only when someone sends
`Command(resume=decisions)` for the same thread_id, even from a new process.
"""

from __future__ import annotations

import json
import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .shop import REQUIRES_APPROVAL, TOOL_SCHEMAS, Shop

SYSTEM_PROMPT = (
    "You are a customer-support operations agent for a small online shop. "
    "Look orders up before acting. Refunds and emails need a human's approval: "
    "request them with the tools and the system will ask the human. "
    "If an action is rejected, do not retry it; tell the user what happened. "
    "Amounts are in naira. Keep answers short."
)


class Decision(TypedDict, total=False):
    action: Literal["approve", "reject", "edit"]
    args: dict[str, Any]  # replacement arguments when action == "edit"
    reason: str


class AgentState(TypedDict):
    # `operator.add` is a reducer: node updates are appended, not overwritten.
    messages: Annotated[list[dict[str, Any]], operator.add]
    audit: Annotated[list[dict[str, Any]], operator.add]
    decisions: dict[str, Decision]  # call_id -> decision for the current batch


def _tool_calls(state: AgentState) -> list[dict[str, Any]]:
    last = state["messages"][-1]
    if last.get("role") != "assistant":
        return []
    return last.get("tool_calls") or []


def pending_actions(state: AgentState) -> list[dict[str, Any]]:
    """The risky tool calls in the latest assistant message, ready to show a human."""
    return [
        {"id": tc["id"], "tool": tc["function"]["name"], "args": json.loads(tc["function"]["arguments"] or "{}")}
        for tc in _tool_calls(state)
        if tc["function"]["name"] in REQUIRES_APPROVAL
    ]


def build_graph(client: Any, model: str, shop: Shop | None = None, checkpointer: Any = None):
    shop = shop or Shop()

    def agent(state: AgentState) -> dict[str, Any]:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *state["messages"]],
            tools=TOOL_SCHEMAS,
        )
        msg = response.choices[0].message
        entry: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            entry["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        return {"messages": [entry], "decisions": {}}

    def route_after_agent(state: AgentState) -> str:
        calls = _tool_calls(state)
        if not calls:
            return END
        if any(tc["function"]["name"] in REQUIRES_APPROVAL for tc in calls):
            return "review"
        return "tools"

    def review(state: AgentState) -> dict[str, Any]:
        # On resume LangGraph re-runs this node from the top and interrupt()
        # returns the human's answer instead of pausing. So: no side effects
        # before this line.
        decisions: dict[str, Decision] = interrupt({"actions": pending_actions(state)})
        return {"decisions": decisions}

    def tools(state: AgentState) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        audit: list[dict[str, Any]] = []
        for tc in _tool_calls(state):
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": json.dumps({"error": "arguments were not valid JSON"})})
                continue

            if name in REQUIRES_APPROVAL:
                # Fail closed: no decision for this call means it does not run.
                decision = state.get("decisions", {}).get(tc["id"], {"action": "reject", "reason": "no decision"})
                action = decision.get("action", "reject")
                if action == "edit":
                    args = decision.get("args", args)
                elif action != "approve":
                    reason = decision.get("reason") or "rejected by reviewer"
                    audit.append({"tool": name, "args": args, "decision": "rejected", "reason": reason})
                    messages.append({"role": "tool", "tool_call_id": tc["id"],
                                     "content": json.dumps({"status": "rejected", "reason": reason})})
                    continue
                audit.append({"tool": name, "args": args, "decision": "edited" if action == "edit" else "approved",
                              "reason": decision.get("reason", "")})

            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": shop.run_tool(name, args)})
        return {"messages": messages, "audit": audit, "decisions": {}}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("review", review)
    graph.add_node("tools", tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, ["review", "tools", END])
    graph.add_edge("review", "tools")
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
