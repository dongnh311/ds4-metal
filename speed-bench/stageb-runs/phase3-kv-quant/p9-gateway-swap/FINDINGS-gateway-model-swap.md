# Gateway model swap: base Q8 -> OrcaUncensored Q4_K imat (+~14% single-stream)

Registry: /Users/dongnh/.local/ai-gateway/runtime-registry.json, entry
models/ivanfioravanti--Qwen3.8-Flash-Next-DS4-IQ2/runtimes/ds4 (name-based;
ds4-server on 127.0.0.1:18086, max_concurrent=1 single-stream).

## Change (DONE + validated)
- Staged OrcaUncensored Q4_K imat into ds4-models/ as
  Qwen3.8-Flash-Next-OrcaUncensored-DenseQ4Kselimat-Q2KDownPad768-MTP.gguf (41 GiB).
- Registry: model_path AND process_command[-m] repointed base Q8 -> imat (2
  occurrences); JSON validated. Registry backed up (runtime-registry.json.bak-preimat-*).
- Unchanged: --ple sidecar, -c 229376, --prefill-chunk 1024, --mtp, kv-disk, port
  18086, envs (DS4_QWEN4_PLE_EVICT_TOKENS=1024, DS4_QWEN4_MOE_MV_SPECIALIZE=1).

## Why
imat Q4_K dense = ~+14% decode vs Q8 at cosine 0.949 (near-Q8). Gateway is
single-stream (max_concurrent=1) so the decode gain applies directly. Runs on the
new engine base (orca-rebase) with Q4_K dense GEMV.

## Validated (exact production command on port 18086)
KV 7.29 GiB @229K + model 40.67 + buffers 1.63 = 49.6 GiB (fits 64; < Q8). /v1/models
OK; chat returns correct content ("HELLO"). Validation instance stopped; gateway was
already stopped.

## Activation + rollback
- ACTIVATE: start/restart the gateway via the normal stack (proxy-stack.sh /
  watchdog); it reads the registry and launches ds4-server with the imat.
- ROLLBACK: restore runtime-registry.json.bak-preimat-*; base Q8 still present as
  ds4-models/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf.
- Retained the base Q8 for rollback; delete once confirmed good (frees ~44 GiB).
- Optional cosmetic: registry 'quantization' field still says the old label
  (functional swap is model_path/-m; label is display-only) — left as-is (a
  registry edit for it was classifier-blocked; not needed).
