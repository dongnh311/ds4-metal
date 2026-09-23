# Qwen3.8 regression gate

Spec: `docs/V41_64GB_BUILD.md` §4. It protects the PROD Qwen3.8 configuration,
read from the gateway registry (`~/.local/ai-gateway/runtime-registry.json`).

| Tier | When | Checks |
| --- | --- | --- |
| fast | every commit touching shared runtime code | `make test-qwen4-kernels test-qwen4-q2`; vi/code replies byte-identical to `baseline/`; registry command unchanged |
| full | end of every phase, before merging to `develop` | fast + decode t/s median of 3 ≥ 97 % of baseline, steady wired ≤ baseline + 0.5 GiB (`vm_stat`), long-context needle found |

Both tiers need the machine free: the PROD gateway's ds4 backend and every
other ds4 process must be stopped, and the user must agree to the run.

```sh
speed-bench/qwen-regression/run.sh fast
speed-bench/qwen-regression/run.sh full
# re-record the reference from the PROD binary (only after the PROD deploy changes):
python3 speed-bench/qwen-regression/qwen_gate.py record --out speed-bench/qwen-regression/baseline --full
```

A failure stops DS4.1 work until it is understood (spec §4).
