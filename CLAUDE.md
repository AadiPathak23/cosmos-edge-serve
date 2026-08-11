# cosmos-edge-serve

FastAPI + Docker inference service for `nvidia/Cosmos-Reason2-2B` (a Qwen3-VL-2B-based
physical-AI reasoning VLM). Takes an image or short video plus a text prompt, returns the
model's reasoning output. The headline deliverable is a benchmark: throughput and p50/p95
latency under concurrency on a T4 GPU.

## Read these first — every session, before doing anything else

1. **`docs/CLAUDE.md`** — the project memory.
   - `STABLE` section: purpose, architecture, key decisions *and the reasoning behind them*,
     conventions, environment. Changes rarely.
   - `CHANGELOG` section: append-only. Never rewrite history.
2. **`docs/PLAN.md`** — phased task breakdown with checkboxes. Current status lives here.

This root file is a stub and is intentionally never expanded. `docs/CLAUDE.md` is the file
that is actually maintained; this one exists only because Claude Code auto-loads
`./CLAUDE.md` from the repo root and nothing else.

## Hard constraints — do not violate without asking

- Total AWS spend must stay under **$10**. Phase 1 is entirely free and local.
- Before **any** step that costs money: state the cost per hour and in total, say whether a
  free alternative exists, then **wait** for confirmation.
- Spot instances only, never on-demand, unless you explain why spot won't work.
- Every EC2 command must be paired with the command that undoes it.
- No paid API, managed service, or AWS service beyond **EC2 + S3** without asking first.
- Never start a phase until the previous one is confirmed working.
- After every meaningful change: append to `docs/CLAUDE.md` CHANGELOG and update the
  checkboxes in `docs/PLAN.md`.
