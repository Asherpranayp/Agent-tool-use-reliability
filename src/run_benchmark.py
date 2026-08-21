"""
Runs every task across a range of injected failure rates AND guardrail
strategies, scores each run against the full failure taxonomy, and writes a
results CSV plus a printed summary table.

Usage:
    python -m src.run_benchmark
    python -m src.run_benchmark --policy heuristic --runs 20
    python -m src.run_benchmark --policy llm --runs 5   # needs ANTHROPIC_API_KEY
"""

import argparse
import csv
import hashlib
import os
import statistics
from collections import defaultdict


def _stable_seed(*parts) -> int:
    """
    Deterministic seed independent of process-level hash randomization.
    Python's built-in hash() on strings is salted per-process by default
    (PYTHONHASHSEED), so using it here would silently break the
    run-to-run reproducibility the benchmark and README rely on.
    """
    key = "|".join(str(p) for p in parts).encode()
    return int(hashlib.md5(key).hexdigest(), 16) % (2**31)

from .agent import run_task
from .guardrails import GUARDRAIL_NAMES, build_strategy
from .tasks import ALLOWED_TOOLS_BY_TASK, TASKS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="heuristic", choices=["heuristic", "llm"])
    parser.add_argument("--runs", type=int, default=15,
                         help="Runs per (task, failure_rate, guardrail) combination")
    parser.add_argument("--failure-rates", type=float, nargs="+",
                         default=[0.0, 0.2, 0.4, 0.6])
    parser.add_argument("--guardrails", nargs="+", default=GUARDRAIL_NAMES,
                         choices=GUARDRAIL_NAMES)
    parser.add_argument("--bug-rate", type=float, default=0.2,
                         help="Probability a run injects a wrong-tool or "
                              "scope-violation attempt at step 0")
    parser.add_argument("--out", default="results.csv")
    args = parser.parse_args()

    rows = []
    summary = defaultdict(list)  # (task_id, failure_rate, guardrail_name) -> row dicts

    run_counter = 0
    for task in TASKS:
        for failure_rate in args.failure_rates:
            for guardrail_name in args.guardrails:
                for run_idx in range(args.runs):
                    run_counter += 1
                    seed = _stable_seed(task.task_id, failure_rate, guardrail_name, run_idx)
                    guardrail = build_strategy(guardrail_name, ALLOWED_TOOLS_BY_TASK)
                    registry = run_task(
                        task, failure_rate, policy_name=args.policy, seed=seed,
                        guardrail=guardrail, bug_rate=args.bug_rate,
                    )
                    score = task.scorer(registry)

                    row = {
                        "task_id": task.task_id,
                        "failure_rate": failure_rate,
                        "guardrail": guardrail_name,
                        "run_idx": run_idx,
                        "num_tool_calls": len(registry.call_log),
                        "cascaded_on_failure": score.get("cascaded_on_failure", False),
                        "redundant_calls": score.get("redundant_calls", 0),
                        "wrong_tool_selected": score.get("wrong_tool_selected", False),
                        "out_of_scope_attempted": score.get("out_of_scope_attempted", False),
                        "out_of_scope_blocked": score.get("out_of_scope_blocked", False),
                    }
                    if score.get("correct_cancel_decision") is not None:
                        row["correct_decision"] = score["correct_cancel_decision"]
                    elif score.get("correct_refund_decision") is not None:
                        row["correct_decision"] = score["correct_refund_decision"]
                    else:
                        row["correct_decision"] = None

                    rows.append(row)
                    summary[(task.task_id, failure_rate, guardrail_name)].append(row)

    fieldnames = ["task_id", "failure_rate", "guardrail", "run_idx", "num_tool_calls",
                  "cascaded_on_failure", "redundant_calls", "wrong_tool_selected",
                  "out_of_scope_attempted", "out_of_scope_blocked", "correct_decision"]
    out_path = os.path.join(os.path.dirname(__file__), "..", args.out)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nRan {run_counter} task executions with policy='{args.policy}'\n")

    by_guardrail = defaultdict(list)
    for row in rows:
        by_guardrail[row["guardrail"]].append(row)

    header = (f"{'guardrail':<16}{'cascade_rate':<14}{'correct_rate':<14}"
              f"{'oos_attempted':<15}{'oos_blocked':<13}{'avg_redundant':<14}")
    print(header)
    print("-" * len(header))
    for guardrail_name in args.guardrails:
        grows = by_guardrail[guardrail_name]
        if not grows:
            continue
        cascade_rate = sum(r["cascaded_on_failure"] for r in grows) / len(grows)
        correct_vals = [r["correct_decision"] for r in grows if r["correct_decision"] is not None]
        correct_rate = (sum(correct_vals) / len(correct_vals)) if correct_vals else float("nan")
        oos_attempted = sum(r["out_of_scope_attempted"] for r in grows)
        oos_blocked = sum(r["out_of_scope_blocked"] for r in grows)
        avg_redundant = statistics.mean(r["redundant_calls"] for r in grows)
        oos_block_rate_str = f"{(oos_blocked / oos_attempted):.0%}" if oos_attempted else "n/a"
        print(f"{guardrail_name:<16}{cascade_rate:<14.2f}{correct_rate:<14.2f}"
              f"{oos_attempted:<15}{oos_block_rate_str:<13}{avg_redundant:<14.2f}")

    print(f"\nFull per-task, per-run results written to: {out_path}")


if __name__ == "__main__":
    main()
