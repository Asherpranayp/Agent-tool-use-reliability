"""
Minimal LangGraph agent wired to the mock tool registry.

Decision-making is pluggable via a `policy` function so the harness can run
in two modes:

  - "heuristic": a small rule-based policy (default, no API key needed) that
    intentionally includes a couple of imperfect recovery behaviors, so the
    benchmark has something realistic to measure.
  - "llm": routes the same decision through a real Claude call if
    ANTHROPIC_API_KEY is set in the environment. Swap this in once you want
    to benchmark an actual model instead of the heuristic stand-in.

This keeps the graph structure (the part worth showing in an interview)
identical regardless of which policy is driving it.
"""

import os
from typing import Any, Dict, List, TypedDict

from langgraph.graph import StateGraph, END

from .mock_tools import ToolRegistry
from .tasks import Task


class AgentState(TypedDict):
    task: Task
    registry: ToolRegistry
    step: int
    max_steps: int
    done: bool
    scratch: Dict[str, Any]


# ---------------------------------------------------------------------------
# Heuristic policy (default - runs with no API key)
# ---------------------------------------------------------------------------

def heuristic_policy(state: AgentState) -> str:
    """
    Decide the next action given task type and what's happened so far.
    Deliberately includes imperfect recovery logic on injected failures,
    so the benchmark has real failure modes to detect rather than a
    hand-tuned policy that never fails.
    """
    task = state["task"]
    registry = state["registry"]
    calls = registry.call_log

    if task.task_id == "cancel_if_late":
        lookup_calls = [c for c in calls if c.tool_name == "lookup_order"]
        cancel_calls = [c for c in calls if c.tool_name == "cancel_order"]

        if not lookup_calls:
            return "lookup_order"

        last_lookup = lookup_calls[-1]
        if last_lookup.was_injected_failure:
            # Imperfect recovery: retries once, then (incorrectly) proceeds
            # to cancel anyway on the second failure - this is the kind of
            # cascade the benchmark is designed to catch.
            if len(lookup_calls) == 1:
                return "lookup_order"  # sensible retry
            elif len(lookup_calls) == 2 and not cancel_calls:
                return "cancel_order"  # BUG (intentional): cascades on repeated failure
            return "stop"

        days = last_lookup.result.get("days_since_order", 0)
        deadline = last_lookup.result.get("cancel_deadline_days", 999)
        if days > deadline and not cancel_calls:
            return "cancel_order"
        return "stop"

    if task.task_id == "refund_after_cancel":
        cancel_calls = [c for c in calls if c.tool_name == "cancel_order"]
        refund_calls = [c for c in calls if c.tool_name == "issue_refund"]

        if not cancel_calls:
            return "cancel_order"

        last_cancel = cancel_calls[-1]
        if last_cancel.was_injected_failure:
            if len(cancel_calls) == 1:
                return "cancel_order"  # sensible retry
            return "stop"  # correct: don't refund a cancellation that never worked

        if last_cancel.result.get("cancelled") and not refund_calls:
            return "issue_refund"
        return "stop"

    return "stop"


# ---------------------------------------------------------------------------
# Real-LLM policy hook (optional, needs ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------

def llm_policy(state: AgentState) -> str:
    """
    Placeholder for routing the same decision through a real Claude call.
    Not wired to a live prompt yet - fill in once you're ready to benchmark
    a real model's tool-selection behavior instead of the heuristic stand-in.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "llm_policy requires ANTHROPIC_API_KEY to be set. "
            "Use policy='heuristic' to run without an API key."
        )
    raise NotImplementedError(
        "TODO: construct a prompt from state['task'] and registry.call_log, "
        "call the Anthropic API, and parse the chosen tool name from the response."
    )


POLICIES = {
    "heuristic": heuristic_policy,
    "llm": llm_policy,
}


# ---------------------------------------------------------------------------
# LangGraph wiring
# ---------------------------------------------------------------------------

def build_graph(policy_name: str = "heuristic"):
    policy = POLICIES[policy_name]

    def decide_and_act(state: AgentState) -> AgentState:
        task = state["task"]
        registry = state["registry"]

        if state["step"] >= state["max_steps"]:
            state["done"] = True
            return state

        action = policy(state)

        if action == "lookup_order":
            registry.lookup_order(task.order_id)
        elif action == "cancel_order":
            registry.cancel_order(task.order_id)
        elif action == "issue_refund":
            registry.issue_refund(task.order_id, amount=49.99)
        elif action == "stop":
            state["done"] = True

        state["step"] += 1
        return state

    def should_continue(state: AgentState) -> str:
        return END if state["done"] else "act"

    graph = StateGraph(AgentState)
    graph.add_node("act", decide_and_act)
    graph.set_entry_point("act")
    graph.add_conditional_edges("act", should_continue, {"act": "act", END: END})
    return graph.compile()


def run_task(task: Task, failure_rate: float, policy_name: str = "heuristic",
             max_steps: int = 6, seed: int = None) -> ToolRegistry:
    registry = ToolRegistry(failure_rate=failure_rate, seed=seed)
    graph = build_graph(policy_name)
    initial_state: AgentState = {
        "task": task,
        "registry": registry,
        "step": 0,
        "max_steps": max_steps,
        "done": False,
        "scratch": {},
    }
    graph.invoke(initial_state)
    return registry
