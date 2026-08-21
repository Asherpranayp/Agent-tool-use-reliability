**Asher Pranay Palle**
PhD Candidate, Artificial Intelligence — Southwest Baptist University | AI Data Engineer, Cardinality AI

## Abstract

LLM agents that call external tools (APIs, databases, retrieval systems) fail in ways that differ structurally from single-turn LLM failures: a wrong tool choice, a malformed argument, an unnecessary call, or a failure to fall back gracefully can cascade through a multi-step task even when each individual model output looks locally reasonable. Production agent frameworks (LangGraph, CrewAI, MCP-based tool-calling) increasingly rely on guardrails — rate limiting, retries, circuit breakers, least-privilege access — to contain these failures operationally, but there is comparatively little standardized methodology for *evaluating* tool-use reliability itself: which failure modes occur, how often, and under what conditions. This proposal outlines a research direction for a tool-use reliability evaluation framework: a taxonomy of agent tool-use failure modes, a benchmark harness for measuring them, and an analysis of which guardrail strategies most effectively contain each failure type.

## 1. Motivation

Building and operating production agent systems surfaces a recurring gap: teams instrument *operational* reliability (latency, uptime, retry counts) far more rigorously than *behavioral* reliability (did the agent choose the right tool, pass correct arguments, recognize when a tool call was unnecessary or unsafe). Guardrail design — least-privilege access, deterministic fallbacks, circuit breakers — is often built reactively, after specific failures are observed in production, rather than validated against a systematic failure taxonomy beforehand. This mirrors a broader gap noted in current agent-evaluation research directions at industry labs (e.g., failure analysis and robustness as explicit research areas in current LLM evaluation postings): the field has stronger tooling for evaluating *what an agent says* than for evaluating *what an agent does*.

## 2. Problem Statement

Given an LLM agent equipped with a fixed set of tools (via MCP-style tool-calling) and a set of representative multi-step tasks, systematically characterize and measure: (a) tool-selection accuracy (right tool for the task), (b) argument-construction correctness, (c) unnecessary or redundant tool invocation, (d) failure-recovery behavior (does the agent retry sensibly, fall back, or cascade the error), and (e) policy/least-privilege violations (does the agent attempt actions outside its granted scope).

## 3. Proposed Approach

- **Failure taxonomy:** Define a structured set of tool-use failure categories (wrong tool, malformed call, redundant call, unsafe/out-of-scope call, poor failure-recovery), grounded in patterns observed in production agent operation.
- **Benchmark harness:** A small set of multi-step synthetic tasks (e.g., "look up X, then update Y if condition Z holds") run against an agent with a fixed MCP-style toolset, instrumented to log every tool call, argument set, and outcome.
- **Guardrail comparison:** Run the same benchmark under different guardrail configurations (no guardrails, retry-only, circuit-breaker, least-privilege-scoped tools) and measure which failure categories each configuration actually contains versus merely delays.
- **Metrics:** Tool-selection accuracy, argument validity rate, redundant-call rate, recovery success rate, and a composite "task completion under failure injection" score — deliberately injecting malformed tool responses to test recovery behavior specifically.

## 4. Expected Contributions

1. A reusable failure taxonomy for LLM agent tool-use, applicable across frameworks (LangGraph, CrewAI, raw MCP tool-calling).
2. A small open benchmark harness for measuring tool-use reliability under failure injection.
3. Empirical comparison of common guardrail strategies against the taxonomy — which strategies contain which failure types, and which don't.

## 5. Relation to Prior Work

This builds on production agent-orchestration and observability practice (distributed tracing, SLO-driven monitoring, circuit breakers) and connects it to the evaluation-methodology side of agent research — an area increasingly named directly in industry research postings (agent evaluation, failure analysis, tool-use evaluation, robustness). The contribution is a bridge between production reliability engineering practice and a more rigorous, benchmarkable evaluation methodology.

## 6. Timeline (Proposed)

| Phase | Duration | Output |
|---|---|---|
| Failure taxonomy + task set design | 1–2 weeks | Defined taxonomy, 8–12 synthetic multi-step tasks |
| Benchmark harness + logging | 2 weeks | Instrumented agent runner with full tool-call tracing |
| Guardrail comparison experiments | 2 weeks | Results across guardrail configurations |
| Write-up and open release | 1 week | Public repo + short technical report |

## 7. Status

Proposed research direction, early implementation. This document will be updated as the benchmark harness and results are produced.

---

*Contact:asherpranay@gmail.com/LinkedIn: https://www.linkedin.com/in/asherpranay/*
