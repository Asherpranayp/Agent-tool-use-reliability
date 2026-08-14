"""
Runs every task across a range of injected failure rates, scores each run,
and writes a results CSV plus a printed summary table.

Usage:
    python -m src.run_benchmark
    python -m src.run_benchmark --policy heuristic --runs 20
"""

import argparse
import csv
import os
import statistics
from collections import defaultdict

from .agent import run_task
from .tasks import TASKS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="heuristic", choices=["heuristic", "llm"])
    parser.add_argument("--runs", type=int, default=15,
                         help="Runs per (task, failure_rate) combination")
    parser.add_argument("--failure-rates", type=float, nargs="+",
                         default=[0.0, 0.2, 0.4, 0.6])
    parser.add_argument("--out", default="results.csv")
    args = parser.parse_args()

    rows = []
    summary = defaultdict(list)  # (task_id, failure_rate) -> list of row dicts

    run_counter = 0
    for task in TASKS:
        for failure_rate in args.failure_rates:
            for run_idx in range(args.runs):
                run_counter += 1
                # Distinct seed per run so failure injection varies across
                # repeated runs, while the whole benchmark stays reproducible
                # end-to-end if you rerun it.
                seed = hash((task.task_id, failure_rate, run_idx)) % (2**31)
                registry = run_task(task, failure_rate, policy_name=args.policy, seed=seed)
                score = task.scorer(registry)

                row = {
                    "task_id": task.task_id,
                    "failure_rate": failure_rate,
                    "run_idx": run_idx,
                    "num_tool_calls": len(registry.call_log),
                    "cascaded_on_failure": score.get("cascaded_on_failure", False),
                    "redundant_calls": score.get("redundant_calls", 0),
                }
                # Merge in task-specific correctness field if present
                for key in ("correct_cancel_decision", "correct_refund_decision"):
                    if key in score:
                        row["correct_decision"] = score[key]

                rows.append(row)
                summary[(task.task_id, failure_rate)].append(row)

    # --- write CSV ---
    fieldnames = ["task_id", "failure_rate", "run_idx", "num_tool_calls",
                  "cascaded_on_failure", "redundant_calls", "correct_decision"]
    out_path = os.path.join(os.path.dirname(__file__), "..", args.out)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # --- printed summary table ---
    print(f"\nRan {run_counter} task executions with policy='{args.policy}'\n")
    header = f"{'task':<20}{'failure_rate':<14}{'cascade_rate':<14}{'correct_rate':<14}{'avg_redundant':<14}"
    print(header)
    print("-" * len(header))

    for (task_id, failure_rate), run_rows in sorted(summary.items()):
        cascade_rate = sum(r["cascaded_on_failure"] for r in run_rows) / len(run_rows)
        correct_vals = [r["correct_decision"] for r in run_rows if r.get("correct_decision") is not None]
        correct_rate = (sum(correct_vals) / len(correct_vals)) if correct_vals else float("nan")
        avg_redundant = statistics.mean(r["redundant_calls"] for r in run_rows)

        print(f"{task_id:<20}{failure_rate:<14}{cascade_rate:<14.2f}{correct_rate:<14.2f}{avg_redundant:<14.2f}")

    print(f"\nFull results written to: {out_path}")


if __name__ == "__main__":
    main()
