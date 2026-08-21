"""
Mock tool implementations for the agent tool-use reliability benchmark.

Each tool simulates a real backend call (order lookup, cancellation, refund,
inventory check) and supports a configurable probability of returning a
malformed / failed response, so the harness can measure how well an agent
recovers from bad tool output rather than just whether it can call tools
when everything works.
"""

import random
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ToolCallRecord:
    tool_name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]
    was_injected_failure: bool


@dataclass
class ToolRegistry:
    """Holds mock tools and a shared call log for the current task run."""
    failure_rate: float = 0.0
    call_log: list = field(default_factory=list)
    seed: int = None
    _rng: random.Random = field(init=False)

    def __post_init__(self):
        # Each run gets its own RNG stream (seeded if a seed was given, for
        # reproducibility; otherwise system-random) so failure injection
        # actually varies across repeated runs instead of replaying the
        # same sequence every time.
        self._rng = random.Random(self.seed)

    def _maybe_inject_failure(self) -> bool:
        return self._rng.random() < self.failure_rate

    def _log(self, tool_name: str, arguments: Dict[str, Any],
              result: Dict[str, Any], failed: bool) -> None:
        self.call_log.append(
            ToolCallRecord(tool_name, arguments, result, failed)
        )

    # ---- mock backend tools -------------------------------------------------

    def lookup_order(self, order_id: str) -> Dict[str, Any]:
        failed = self._maybe_inject_failure()
        if failed:
            # Malformed response: missing expected fields
            result = {"error": "malformed_response", "order_id": order_id}
        else:
            result = {
                "order_id": order_id,
                "status": self._rng.choice(["shipped", "processing", "delayed"]),
                "days_since_order": self._rng.randint(1, 20),
                "cancel_deadline_days": 7,
            }
        self._log("lookup_order", {"order_id": order_id}, result, failed)
        return result

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        failed = self._maybe_inject_failure()
        if failed:
            result = {"error": "service_unavailable", "order_id": order_id}
        else:
            result = {"order_id": order_id, "cancelled": True}
        self._log("cancel_order", {"order_id": order_id}, result, failed)
        return result

    def check_inventory(self, sku: str) -> Dict[str, Any]:
        failed = self._maybe_inject_failure()
        if failed:
            result = {"error": "timeout", "sku": sku}
        else:
            result = {"sku": sku, "in_stock": self._rng.choice([True, False]),
                       "quantity": self._rng.randint(0, 50)}
        self._log("check_inventory", {"sku": sku}, result, failed)
        return result

    def issue_refund(self, order_id: str, amount: float) -> Dict[str, Any]:
        failed = self._maybe_inject_failure()
        if failed:
            result = {"error": "malformed_response", "order_id": order_id}
        else:
            result = {"order_id": order_id, "amount": amount, "refunded": True}
        self._log("issue_refund", {"order_id": order_id, "amount": amount},
                   result, failed)
        return result

    def reset(self, failure_rate: float = None) -> None:
        self.call_log = []
        if failure_rate is not None:
            self.failure_rate = failure_rate

    # ---- generic dispatcher + guardrail support ------------------------------

    def call(self, tool_name: str, **arguments) -> Dict[str, Any]:
        """
        Dispatch by tool name so guardrail strategies can wrap any tool call
        uniformly without knowing individual method signatures.
        """
        method = getattr(self, tool_name, None)
        if method is None:
            result = {"error": "unknown_tool", "tool": tool_name}
            self._log(tool_name, arguments, result, failed=True)
            return result
        return method(**arguments)

    def record_blocked_call(self, tool_name: str, arguments: Dict[str, Any],
                              reason: str) -> Dict[str, Any]:
        """
        Log a call a guardrail prevented from ever reaching the backend
        (circuit open, out-of-scope, etc.), so the benchmark can distinguish
        "guardrail prevented this" from "agent never attempted this."
        """
        result = {"error": reason, "tool": tool_name, "guardrail_blocked": True}
        self._log(tool_name, arguments, result, failed=True)
        return result

    # ---- generic dispatcher, used by the guardrail layer --------------------


