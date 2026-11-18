"""Tool tests: plain functions, no LLM involved."""

from __future__ import annotations

import json

from agent import REQUIRES_APPROVAL, Shop


def test_lookup_order_is_case_insensitive():
    assert Shop().lookup_order("a1042")["customer"] == "Amaka Eze"


def test_lookup_unknown_order():
    assert "error" in Shop().lookup_order("Z9999")


def test_find_orders_by_email():
    assert Shop().find_orders("TUNDE@example.com")["orders"] == ["A1037"]


def test_refund_updates_balance_and_ids():
    shop = Shop()
    first = shop.issue_refund("A1042", 5000, "late")
    second = shop.issue_refund("A1042", 3500, "late")
    assert first["refund_id"] == "RF-0001" and second["refund_id"] == "RF-0002"
    assert shop.orders["A1042"]["refunded"] == 8500


def test_refund_cannot_exceed_balance():
    shop = Shop()
    assert "error" in shop.issue_refund("A1029", 100, "already refunded in full")
    assert "error" in shop.issue_refund("A1042", 99999, "too much")
    assert "error" in shop.issue_refund("A1042", -5, "negative")
    assert shop.refunds == []


def test_send_email_validates_address():
    shop = Shop()
    assert "error" in shop.send_email("not-an-email", "hi", "body")
    assert shop.send_email("a@b.co", "hi", "body")["status"] == "sent"
    assert len(shop.outbox) == 1


def test_run_tool_handles_bad_input():
    shop = Shop()
    assert "unknown tool" in shop.run_tool("drop_tables", {})
    assert "bad arguments" in json.loads(shop.run_tool("lookup_order", {"nope": 1}))["error"]


def test_seed_data_is_not_shared_between_shops():
    a, b = Shop(), Shop()
    a.issue_refund("A1042", 1000, "x")
    assert b.orders["A1042"]["refunded"] == 0


def test_only_write_actions_need_approval():
    assert REQUIRES_APPROVAL == {"issue_refund", "send_email"}
