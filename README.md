**Asher Pranay Palle**
PhD Candidate, Artificial Intelligence — Southwest Baptist University | AI Data Engineer, Cardinality AI

## Abstract

LLM agents that call external tools (APIs, databases, retrieval systems) fail in ways that differ structurally from single-turn LLM failures: a wrong tool choice, a malformed argument, an unnecessary call, or a failure to fall back gracefully can cascade through a multi-step task even when each individual model output looks locally reasonable. Production agent frameworks (LangGraph, CrewAI, MCP-based tool-calling) increasingly rely on guardrails — rate limiting, retries, circuit breakers, least-privilege access — to contain these failures operationally, but there is comparatively little standardized methodology for *evaluating* tool-use reliability itself: which failure modes occur, how often, and under what conditions. This proposal outlines a research direction for a tool-use reliability evaluation framework: a taxonomy of agent tool-use failure modes, a benchmark harness for measuring them, and an analysis of which guardrail strategies most effectively contain each failure type.

## 1. Motivation

Building and operating production agent systems surfaces a recurring gap: teams instrument *operational* reliability (latency, uptime, retry counts) far more rigorously than *behavioral* reliability (did the agent choose the right tool, pass correct arguments, recognize when a tool call was unnecessary or unsafe). Guardrail design — least-privilege access, deterministic fallbacks, circuit breakers — is often built reactively, after specific failures are observed in production, rather than validated against a systematic failure taxonomy beforehand. This mirrors a broader gap noted in current agent-evaluation research directions at industry labs (e.g., failure analysis and robustness as explicit research areas in current LLM evaluation postings): the field has stronger tooling for evaluating *what an agent says* than for evaluating *what an agent does*.

## 2. Problem Statement

Given an LLM agent equipped with a fixed set of tools (via MCP-style tool-calling) and a set of representative multi-step tasks, systematically characterize and measure: (a) tool-selection accuracy (right tool for the task), (b) argument-construction correctness, (c) unnecessary or redundant tool invocation, (d) failure-recovery behavior (does the agent retry sensibly, fall back, or cascade the error), and (e) policy/least-privilege violations (does the agent attempt actions outside its granted scope).

## 3. Approach

- **Failure taxonomy:** A structured set of tool-use failure categories — wrong tool selection, malformed/redundant calls, unsafe/out-of-scope calls, poor failure-recovery — grounded in patterns observed in production agent operation, and each one directly instrumented and measured by the harness's scorers (`src/tasks.py`).
- **Benchmark harness:** Four multi-step synthetic tasks (`src/tasks.py`) run against a LangGraph-wired agent (`src/agent.py`) with a fixed mock toolset (`src/mock_tools.py`), instrumented to log every tool call, argument set, and outcome — including calls a guardrail blocked before they ever reached the backend.
- **Guardrail comparison:** `src/guardrails.py` implements all four strategies named below — `none`, `retry_only`, `circuit_breaker`, `least_privilege` — as a call-wrapping layer independent of the policy, so the same task + policy combination re-runs identically under each strategy and is scored on the same metrics.
- **Metrics:** correct-decision rate, cascade-on-failure rate, redundant-call count, wrong-tool-selection rate, and out-of-scope-attempt / out-of-scope-blocked rate per guardrail configuration — deliberately injecting malformed tool responses plus occasional (seeded, reproducible) wrong-tool and scope-violation attempts to exercise every category above.

## 4. Expected Contributions

1. A reusable failure taxonomy for LLM agent tool-use, applicable across frameworks (LangGraph, CrewAI, raw MCP tool-calling).
2. A small open benchmark harness for measuring tool-use reliability under failure injection.
3. Empirical comparison of common guardrail strategies against the taxonomy — which strategies contain which failure types, and which don't.

## 4b. Results

3,200 task executions (4 tasks × 4 failure rates × 4 guardrail configurations × 50 runs), `heuristic` policy, `bug_rate=0.2`. Full per-run data in [`results.csv`](results.csv); reproduce with `python -m src.run_benchmark --runs 50`.

| Guardrail | Cascade rate | Correct-decision rate | Out-of-scope attempted | Out-of-scope blocked | Avg. redundant calls |
|---|---|---|---|---|---|
| none | 0.13 | 0.90 | 83 | 0% | 0.00 |
| retry_only | 0.12 | 0.91 | 81 | 0% | 0.00 |
| circuit_breaker | 0.12 | 0.91 | 76 | 0% | 0.00 |
| least_privilege | 0.11 | 0.91 | 69 | **100%** | 0.00 |

**Finding:** `least_privilege` is the only strategy that contains unsafe/out-of-scope tool calls — it blocks 100% of them, consistently across all four tasks, because it checks the call against an allow-list before it ever reaches the backend. `retry_only` and `circuit_breaker` block 0% of them, because they operate on individual tool-call *outcomes* (did this call fail?), not on whether the call was authorized in the first place. Neither strategy meaningfully reduces the cascade rate either — cascades in this harness originate from the policy making a bad *decision* after a failed lookup, not from a tool itself misbehaving, so a call-level guardrail can contain a failing tool but can't correct a flawed decision built on top of it. That's the concrete version of the taxonomy's distinction between failure modes a guardrail *contains* versus one it merely *delays*: circuit-breaker delays/limits repeated failure on a single tool, but doesn't stop a policy from acting on incomplete information once that tool has already failed once or twice.

## 5. Relation to Prior Work

This builds on production agent-orchestration and observability practice (distributed tracing, SLO-driven monitoring, circuit breakers) and connects it to the evaluation-methodology side of agent research — an area increasingly named directly in industry research postings (agent evaluation, failure analysis, tool-use evaluation, robustness). The contribution is a bridge between production reliability engineering practice and a more rigorous, benchmarkable evaluation methodology.

## 6. Status and Next Steps

| Phase | Status | Output |
|---|---|---|
| Failure taxonomy + task set design | Done (4 of a planned 8–12 tasks) | Taxonomy covering all 5 failure categories; `src/tasks.py` |
| Benchmark harness + logging | Done | LangGraph agent, guardrail-wrapping call layer, full tool-call tracing (`src/agent.py`, `src/guardrails.py`, `src/mock_tools.py`) |
| Guardrail comparison experiments | Done (heuristic policy) | Results table above, `results.csv` |
| Real-LLM policy benchmarking | Implemented, not yet run | `src/agent.py`'s `llm_policy` calls Claude with forced tool use; needs `ANTHROPIC_API_KEY` and a benchmark pass to compare against the heuristic baseline |
| Expanded task set (8–12) | Not started | Extend `src/tasks.py` following the existing task/scorer pattern |
| Write-up | This README | — |

**Next planned step:** run the same 3,200-execution sweep with `--policy llm` against Claude and GPT-family models, to compare real model tool-selection and recovery behavior against the heuristic baseline above rather than only against a hand-scripted policy.

---

*Contact: asherpranay@gmail.com · [LinkedIn](https://www.linkedin.com/in/asherpranay/)*
