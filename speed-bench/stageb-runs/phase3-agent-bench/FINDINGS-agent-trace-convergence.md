# Agent trace to convergence (opt-in follow-up) — the IQ2 model's autonomous-planning limit

Follow-up to the throughput bench: drive ds4-server (--mtp, thinking on) through a
coding task with a workflow-directed system prompt (inspect -> edit -> test -> done)
and tool feedback that guides the fix, to capture a trace that RUNS TO CONVERGENCE
(model writes the fix, tests pass, model stops and summarizes). Harness:
agent_trace.py; transcripts agent_trace_run{1,2}.json.

## Result — the model stops but does NOT complete the fix (2/2 runs)
| run | max_tokens | temp | turns | tool sequence | write_file | fix_applied | converged |
|-----|-----------|------|-------|---------------|-----------:|------------:|-----------|
| 1 | 500  | 0.2 | 10 | list, read×7, run_tests, (final) | 0 | NO | yes (stopped) |
| 2 | 1200 | 0.0 | 10 | list, read×6, run_tests, read, (final) | 0 | NO | yes (stopped) |

Both runs: the model inspects thoroughly (1 list + ~7 reads incl. login.py, whose
source is annotated with the bug), runs the tests once (sees the KeyError failure),
then **stops calling tools and emits a final turn WITHOUT ever calling write_file**
— i.e. it declares done without applying the edit. Not a token-budget artifact:
run 2 (1200 tokens, temp 0) behaves identically to run 1 (500 tokens). Thinking is
on (`reasoning_content` present); the model reasons each turn but does not take the
write action.

## Mechanics are healthy (throughput is not the problem)
- tok/s(wall): mean ~31-32, max ~38 (≈ MTP decode ceiling on generation-heavy turns).
- prefix KV cache: 91-99% hit as context grows (541 -> ~2100 tokens).
- swap growth: 0.0 MiB; well-formed, correctly-typed tool_calls throughout; no errors.

## Interpretation
The light Orca model (IQ2 experts) has solid agentic *mechanics* (fast, cheap
context reuse, valid tool schemas) but limited autonomous *planning/execution* in a
bare tool loop: it over-inspects and fails to commit the edit. This is consistent
with heavy quantization trading long-horizon agentic reliability for size/speed,
and with the HF field usage — people run it under OpenCode, whose harness adds
planning scaffolding (explicit edit steps, retries, diffs) that a bare loop lacks.
It is NOT a runtime/throughput regression. For autonomous coding, either a stronger
agent scaffold or a higher-precision model is needed; for throughput-bound assisted
use the model performs well.

## Caveat
Synthetic tool feedback + one task; a broader trace suite (varied tasks, a real
OpenCode session capture) would strengthen the conclusion. Directly relevant to the
dense-quant experiment: lowering precision further (dense Q4_0) is expected to
reduce agentic planning further, not improve it — the quant tradeoff is quality.

## Receipts
agent_trace.py, agent_trace_run1.json (max500), agent_trace_run2.json (max1200/temp0),
agent_trace_run2.log.
