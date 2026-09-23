# V4.1 Phase 0(b) tools

Spec: `docs/V41_64GB_BUILD.md` §7. Plan: `docs/superpowers/plans/2026-09-23-v41-phase0b-measure.md`.

| Tool | Purpose |
| --- | --- |
| `gguf_bytes.py MODEL` | per-token byte accounting and byte-floor roofline from the GGUF header |
| `router_locality.py LOG --bytes bytes.json` | LRU hit rate vs cache budget, next-token overlap, from `DS4_V41_ROUTER_LOG` |
| `phase0.py prompts/run/table/report` | run the speed and locality plans with `ds4-bench` and summarize |

Runtime diagnostics (off by default, outputs unchanged):
`DS4_V41_ROUTER_LOG=<path>`, `DS4_V41_DECODE_PROFILE=1` (with
`DS4_METAL_GPU_BUSY_PROFILE=1`), plus the existing
`DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1`.

Runs need the machine free (no other ds4 process, the user's go-ahead). Raw
output goes to `~/orca/workspaces/ds4-metal-data/v41-phase0/<YYYYMMDD>/`; only
`results.csv`, `bytes.json`, the locality JSONs and `RESULTS.md` are committed,
under `speed-bench/v41/phase0-<YYYYMMDD>/`.
