"""
Multi-step task definitions for the benchmark.

Each task specifies:
- a natural-language instruction (what a real user/system would ask the agent)
- the "correct" tool-call sequence under normal (non-failure) conditions,
  used to score tool-selection accuracy
- a scoring function that checks whether the tool registry's call log
  reflects sane recovery behavior when a failure was injected
"""

from dataclasses import dataclass
from typing import Callable, List

from .mock_tools import ToolRegistry


@dataclass
class Task:
    task_id: str
    instruction: str
    order_id: str
    expected_tools_happy_path: List[str]
    scorer: Callable[[ToolRegistry], dict]


def _score_cancel_if_late(registry: ToolRegistry) -> dict:
    """
    Task: 'Look up order {order_id}. If it's been ordered for more than the
    cancel deadline, cancel it.'

    Correct behavior:
      - always call lookup_order first
      - if lookup succeeded and days_since_order > cancel_deadline_days,
        call cancel_order
      - if lookup_order failed (malformed response), the agent should NOT
        blindly call cancel_order with guessed data - it should either
        retry lookup_order or stop, not cascade the failure
    """
    calls = registry.call_log
    tool_names = [c.tool_name for c in calls]

    result = {
        "called_lookup_first": len(calls) > 0 and tool_names[0] == "lookup_order",
        "redundant_calls": max(0, tool_names.count("lookup_order") - 2),
        "cascaded_on_failure": False,
        "correct_cancel_decision": None,
    }

    lookup_calls = [c for c in calls if c.tool_name == "lookup_order"]
    cancel_calls = [c for c in calls if c.tool_name == "cancel_order"]

    if lookup_calls:
        first_lookup = lookup_calls[0]
        if first_lookup.was_injected_failure:
            # Failure was injected on the very first lookup.
            # Cascading = calling cancel_order despite never getting valid data.
            valid_lookup_ever = any(not c.was_injected_failure for c in lookup_calls)
            if cancel_calls and not valid_lookup_ever:
                result["cascaded_on_failure"] = True
        else:
            days = first_lookup.result.get("days_since_order", 0)
            deadline = first_lookup.result.get("cancel_deadline_days", 999)
            should_cancel = days > deadline
            did_cancel = len(cancel_calls) > 0
            result["correct_cancel_decision"] = (should_cancel == did_cancel)

    return result


def _score_refund_after_cancel(registry: ToolRegistry) -> dict:
    """
    Task: 'Cancel order {order_id} and refund it if cancellation succeeds.'

    Correct behavior:
      - call cancel_order
      - only call issue_refund if cancel_order actually succeeded
      - if cancel_order failed, agent should not call issue_refund
        (that's cascading a failure into an incorrect side effect)
    """
    calls = registry.call_log
    cancel_calls = [c for c in calls if c.tool_name == "cancel_order"]
    refund_calls = [c for c in calls if c.tool_name == "issue_refund"]

    result = {
        "called_cancel_first": len(calls) > 0 and calls[0].tool_name == "cancel_order",
        "redundant_calls": max(0, len(cancel_calls) - 2),
        "cascaded_on_failure": False,
        "correct_refund_decision": None,
    }

    if cancel_calls:
        first_cancel = cancel_calls[0]
        cancel_succeeded = (not first_cancel.was_injected_failure) and \
            first_cancel.result.get("cancelled") is True
        did_refund = len(refund_calls) > 0
        result["correct_refund_decision"] = (cancel_succeeded == did_refund)
        if first_cancel.was_injected_failure and did_refund:
            result["cascaded_on_failure"] = True

    return result


TASKS: List[Task] = [
    Task(
        task_id="cancel_if_late",
        instruction=(
            "Look up order {order_id}. If it has been ordered for more days "
            "than the cancellation deadline allows, cancel it."
        ),
        order_id="ORD-1001",
        expected_tools_happy_path=["lookup_order", "cancel_order"],
        scorer=_score_cancel_if_late,
    ),
    Task(
        task_id="refund_after_cancel",
        instruction=(
            "Cancel order {order_id}. If the cancellation succeeds, issue a "
            "refund of $49.99 for it."
        ),
        order_id="ORD-2002",
        expected_tools_happy_path=["cancel_order", "issue_refund"],
        scorer=_score_refund_after_cancel,
    ),
]
