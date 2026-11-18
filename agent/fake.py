"""Fake LLM clients: no network, no API key.

- ScriptedClient replays a fixed list of replies (used in unit tests).
- DemoClient reads the conversation and plays a few realistic scripted
  support conversations (used by tests AND by demo mode).

Both mimic the shape of `openai.OpenAI().chat.completions.create(...)`.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace as NS
from typing import Any, Iterable


def tool_call(call_id: str, name: str, args: dict[str, Any]) -> NS:
    return NS(id=call_id, type="function", function=NS(name=name, arguments=json.dumps(args)))


def reply(content: str | None = None, tool_calls: list[NS] | None = None) -> NS:
    return NS(choices=[NS(message=NS(content=content, tool_calls=tool_calls))])


class ScriptedClient:
    """Returns the given replies in order and records every request."""

    def __init__(self, replies: Iterable[NS]):
        self._replies = iter(replies)
        self.calls: list[dict[str, Any]] = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs: Any) -> NS:
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return next(self._replies)


def naira(amount: float) -> str:
    return f"₦{amount:,.0f}"


class DemoClient:
    """Rule-based stand-in for a model, so the whole flow runs offline."""

    ORDER_RE = re.compile(r"\b[Aa]\d{4}\b")
    EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
    AMOUNT_RE = re.compile(r"(?:₦|NGN|N)\s?([\d,]+)")

    def __init__(self) -> None:
        self.calls = 0
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, *, messages: list[dict[str, Any]], **_: Any) -> NS:
        self.calls += 1
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        text: str = messages[last_user]["content"]
        results = [json.loads(m["content"]) for m in messages[last_user + 1:] if m["role"] == "tool"]
        cid = f"call_{len(messages)}"

        order_match = self.ORDER_RE.search(text)
        email_match = self.EMAIL_RE.search(text)

        if "refund" in text.lower():
            return self._refund(text, (order_match.group(0) if order_match else "A1042").upper(), results, cid)
        if order_match:
            return self._status(order_match.group(0).upper(), results, cid)
        if email_match:
            return self._find(email_match.group(0), results, cid)
        return reply(
            "I can look up orders (try 'Where is order A1037?'), find a customer's orders by email, "
            "and refund or email customers once you approve. Try: 'Refund order A1042, the mugs arrived broken.'"
        )

    # --- scripted conversations ---------------------------------------------

    def _status(self, order_id: str, results: list[dict[str, Any]], cid: str) -> NS:
        if not results:
            return reply(tool_calls=[tool_call(cid, "lookup_order", {"order_id": order_id})])
        o = results[-1]
        if "error" in o:
            return reply(f"I couldn't find order {order_id}.")
        eta = f", expected {o['eta']}" if o.get("eta") else ""
        return reply(f"Order {o['id']} ({o['item']}, {naira(o['total'])}) is {o['status']}{eta}.")

    def _find(self, email: str, results: list[dict[str, Any]], cid: str) -> NS:
        if not results:
            return reply(tool_calls=[tool_call(cid, "find_orders", {"email": email})])
        ids = results[-1].get("orders", [])
        return reply(f"{email} has {len(ids)} order(s): {', '.join(ids)}." if ids else f"No orders found for {email}.")

    def _refund(self, text: str, order_id: str, results: list[dict[str, Any]], cid: str) -> NS:
        if not results:
            return reply(tool_calls=[tool_call(cid, "lookup_order", {"order_id": order_id})])

        if len(results) == 1:
            o = results[0]
            if "error" in o:
                return reply(f"I couldn't find order {order_id}, so I haven't refunded anything.")
            remaining = o["total"] - o["refunded"]
            if remaining <= 0:
                return reply(f"Order {o['id']} was already refunded in full ({naira(o['refunded'])}). Nothing to do.")
            m = self.AMOUNT_RE.search(text)
            amount = min(float(m.group(1).replace(",", "")), remaining) if m else remaining
            reason = "Item arrived damaged" if re.search(r"broke|damag|crack", text, re.I) else "Customer request"
            first_name = o["customer"].split()[0]
            return reply(
                f"Order {o['id']} is {o['status']}. I'll refund {naira(amount)} and let {first_name} know.",
                tool_calls=[
                    tool_call(f"{cid}_a", "issue_refund", {"order_id": o["id"], "amount": amount, "reason": reason}),
                    tool_call(f"{cid}_b", "send_email", {
                        "to": o["email"],
                        "subject": f"Your refund for order {o['id']}",
                        "body": f"Hi {first_name}, sorry about the problem with your {o['item']}. "
                                f"We've refunded {naira(amount)}; it should reach you in 3-5 working days.",
                    }),
                ],
            )

        refund, email = results[1], results[2] if len(results) > 2 else {}
        lines = []
        if refund.get("status") == "refunded":
            lines.append(f"Refunded {naira(refund['amount'])} on {refund['order_id']} ({refund['refund_id']}).")
        elif refund.get("status") == "rejected":
            lines.append(f"The refund was not approved ({refund['reason']}), so no money moved.")
        else:
            lines.append(f"The refund failed: {refund.get('error', 'unknown error')}.")
        if email.get("status") == "sent":
            lines.append(f"Emailed {email['to']}.")
        elif email.get("status") == "rejected":
            lines.append("The customer email was not sent.")
        return reply(" ".join(lines))
