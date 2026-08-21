"""
Guardrail strategies for the agent tool-use reliability benchmark.

Each strategy wraps *how a tool call reaches the mock backend*, independent
of the policy that decided to make the call. This lets the harness run the
identical task + policy combination under different guardrail configurations
and measure which strategy actually contains which failure category, versus
merely delaying it - the comparison the README's "Proposed Approach" names
but that wasn't previously implemented anywhere in the code.

Strategies:
  - none            : baseline, calls pass through unmodified
  - retry_only       : retries a failed call once before giving up
  - circuit_breaker  : after N consecutive failures on a tool, "opens" the
                       circuit and blocks further calls to that tool for the
                       rest of the task run
  - least_privilege  : blocks calls to tools outside an explicit allow-list
                       for the current task, before the call ever reaches
                       the backend
"""

from collections import defaultdict
from typing import Any, Dict, Set


class GuardrailStrategy:
    """
    Base class. wrap_call receives the live ToolRegistry so it can both
    dispatch calls through registry.call(...) and log blocked/short-circuited
    attempts via registry.record_blocked_call(...) - blocked attempts must
    still land in the call log, or the benchmark can't measure whether a
    guardrail actually prevented a failure versus the agent simply never
    attempting it.
    """
    name = "none"

    def wrap_call(self, tool_name: str, arguments: Dict[str, Any],
                   registry, task_id: str) -> Dict[str, Any]:
        return registry.call(tool_name, **arguments)

    def reset(self) -> None:
        """Called at the start of each task run so state doesn't leak across runs."""
        pass


class NoGuardrailStrategy(GuardrailStrategy):
    name = "none"


class RetryOnlyStrategy(GuardrailStrategy):
    name = "retry_only"

    def __init__(self, max_retries: int = 1):
        self.max_retries = max_retries

    def wrap_call(self, tool_name, arguments, registry, task_id):
        result = registry.call(tool_name, **arguments)
        retries = 0
        while result.get("error") and retries < self.max_retries:
            result = registry.call(tool_name, **arguments)
            retries += 1
        return result


class CircuitBreakerStrategy(GuardrailStrategy):
    name = "circuit_breaker"

    def __init__(self, failure_threshold: int = 2):
        self.failure_threshold = failure_threshold
        self._consecutive_failures = defaultdict(int)
        self._open = defaultdict(bool)

    def reset(self):
        self._consecutive_failures = defaultdict(int)
        self._open = defaultdict(bool)

    def wrap_call(self, tool_name, arguments, registry, task_id):
        if self._open[tool_name]:
            return registry.record_blocked_call(tool_name, arguments, "circuit_open")

        result = registry.call(tool_name, **arguments)
        if result.get("error"):
            self._consecutive_failures[tool_name] += 1
            if self._consecutive_failures[tool_name] >= self.failure_threshold:
                self._open[tool_name] = True
        else:
            self._consecutive_failures[tool_name] = 0
        return result


class LeastPrivilegeStrategy(GuardrailStrategy):
    name = "least_privilege"

    def __init__(self, allowed_tools_by_task: Dict[str, Set[str]]):
        self.allowed_tools_by_task = allowed_tools_by_task

    def wrap_call(self, tool_name, arguments, registry, task_id):
        allowed = self.allowed_tools_by_task.get(task_id, set())
        if tool_name not in allowed:
            return registry.record_blocked_call(tool_name, arguments, "policy_violation_out_of_scope")
        return registry.call(tool_name, **arguments)


def build_strategy(name: str, allowed_tools_by_task: Dict[str, Set[str]] = None) -> GuardrailStrategy:
    if name == "none":
        return NoGuardrailStrategy()
    if name == "retry_only":
        return RetryOnlyStrategy()
    if name == "circuit_breaker":
        return CircuitBreakerStrategy()
    if name == "least_privilege":
        return LeastPrivilegeStrategy(allowed_tools_by_task or {})
    raise ValueError(f"Unknown guardrail strategy: {name}")


GUARDRAIL_NAMES = ["none", "retry_only", "circuit_breaker", "least_privilege"]
