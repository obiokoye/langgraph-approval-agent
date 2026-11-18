"""A tiny fake online shop and the tools the agent can use on it.

Read-only tools (look things up) run straight away. Tools that change
something or contact a customer are listed in REQUIRES_APPROVAL, and the
graph pauses for a human before they run.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

SEED_ORDERS: dict[str, dict[str, Any]] = {
    "A1042": {
        "id": "A1042",
        "customer": "Amaka Eze",
        "email": "amaka@example.com",
        "item": "Ceramic mug set (4)",
        "total": 18500,
        "status": "delivered",
        "refunded": 0,
    },
    "A1037": {
        "id": "A1037",
        "customer": "Tunde Bello",
        "email": "tunde@example.com",
        "item": "Wireless keyboard",
        "total": 42000,
        "status": "shipped",
        "eta": "2026-11-20",
        "refunded": 0,
    },
    "A1029": {
        "id": "A1029",
        "customer": "Chioma Obi",
        "email": "chioma@example.com",
        "item": "Desk lamp",
        "total": 15000,
        "status": "delivered",
        "refunded": 15000,
    },
}


@dataclass
class Shop:
    """In-memory store. Each run starts from the same seed data."""

    orders: dict[str, dict[str, Any]] = field(default_factory=lambda: deepcopy(SEED_ORDERS))
    refunds: list[dict[str, Any]] = field(default_factory=list)
    outbox: list[dict[str, Any]] = field(default_factory=list)

    # --- read-only tools ------------------------------------------------------

    def lookup_order(self, order_id: str) -> dict[str, Any]:
        order = self.orders.get(order_id.strip().upper())
        if order is None:
            return {"error": f"order '{order_id}' not found"}
        return dict(order)

    def find_orders(self, email: str) -> dict[str, Any]:
        found = [o["id"] for o in self.orders.values() if o["email"].lower() == email.strip().lower()]
        return {"email": email, "orders": found}

    # --- actions that need approval ------------------------------------------

    def issue_refund(self, order_id: str, amount: float, reason: str) -> dict[str, Any]:
        order = self.orders.get(order_id.strip().upper())
        if order is None:
            return {"error": f"order '{order_id}' not found"}
        if amount <= 0:
            return {"error": "amount must be positive"}
        remaining = order["total"] - order["refunded"]
        if amount > remaining:
            return {"error": f"amount exceeds refundable balance of {remaining}"}
        order["refunded"] += amount
        refund = {"refund_id": f"RF-{len(self.refunds) + 1:04d}", "order_id": order["id"], "amount": amount, "reason": reason}
        self.refunds.append(refund)
        return {"status": "refunded", **refund}

    def send_email(self, to: str, subject: str, body: str) -> dict[str, Any]:
        if "@" not in to:
            return {"error": f"invalid email address '{to}'"}
        self.outbox.append({"to": to, "subject": subject, "body": body})
        return {"status": "sent", "to": to, "subject": subject}

    # --- dispatcher -----------------------------------------------------------

    def tools(self) -> dict[str, Callable[..., dict[str, Any]]]:
        return {
            "lookup_order": self.lookup_order,
            "find_orders": self.find_orders,
            "issue_refund": self.issue_refund,
            "send_email": self.send_email,
        }

    def run_tool(self, name: str, args: dict[str, Any]) -> str:
        """Run a tool by name; always returns a JSON string (errors included)."""
        func = self.tools().get(name)
        if func is None:
            return json.dumps({"error": f"unknown tool '{name}'"})
        try:
            return json.dumps(func(**args))
        except TypeError as exc:
            return json.dumps({"error": f"bad arguments: {exc}"})


# Tools that change data or reach a customer: a human must approve them first.
REQUIRES_APPROVAL: set[str] = {"issue_refund", "send_email"}


def _fn(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOL_SCHEMAS: list[dict[str, Any]] = [
    _fn("lookup_order", "Get an order's details, status and refunded amount.",
        {"order_id": {"type": "string", "description": "e.g. 'A1042'"}}, ["order_id"]),
    _fn("find_orders", "List order IDs for a customer email.",
        {"email": {"type": "string"}}, ["email"]),
    _fn("issue_refund", "Refund money on an order (amount in naira). Needs human approval.",
        {
            "order_id": {"type": "string"},
            "amount": {"type": "number", "description": "Naira, e.g. 18500"},
            "reason": {"type": "string"},
        },
        ["order_id", "amount", "reason"]),
    _fn("send_email", "Email a customer. Needs human approval.",
        {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
        ["to", "subject", "body"]),
]
