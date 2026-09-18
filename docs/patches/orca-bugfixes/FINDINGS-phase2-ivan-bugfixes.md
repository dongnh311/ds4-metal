# Phase 2 — Fix Ivan's bugs B1–B4 (prod codebase)

Confirmed-plan Phase 2: fix the four known bugs in the model runtime. Fixes were
made in the PROD checkout `~/.local/share/ai-gateway/ds4-metal` (ivanfioravanti
clone, branch `qwen3.8-flash-next` @ 6c1e836 → local branch **`orca-bugfixes`**),
which is the code that actually runs the model; the exact diffs are preserved as
patches beside this file (0001 B1, 0002 B4, 0003 B2/B3). B1 and B2/B3 were ported
from dongnh311/scallop (which already had them); B4 was new in both trees.

## B1 — image-cache SIGSEGV (ds4_server.c, +5/-1) — commit 73c08a6
`server_encode_image` told `server_image_cache_put` every vision row was 4096
floats wide, but the Qwen3.8 encoder allocates DS4_N_EMBD (2560) wide rows →
cache memcpy over-read (4096−2560)·tokens·4 bytes past the source → SIGSEGV on
larger images. Fix: use `ds4_engine_embd_dim(s->engine)` (existing accessor).
Compiles clean; normal (non-vision) generation unaffected. (Full vision repro
needs an mmproj not set up here; fix is the identical, validated scallop change.)

## B4 — Metal kernels loaded CWD-relative → garbage (ds4_metal.m, +38) — 3c63553
`ds4_gpu_full_source` resolved every `metal/*.metal` only against the process
CWD, so running the binary from a dir with a stale/mismatched `./metal` silently
compiled the wrong kernels → garbage output (this was the root of the earlier
"We need!!!!" prod runs and the whole "run from its own dir" workaround). Fix:
resolve the kernel dir relative to the executable (`_NSGetExecutablePath` +
realpath + dirname) FIRST, add a `DS4_METAL_DIR` diagnostic env, keep CWD as
fallback, log the chosen dir once. No new user flag.
**Validated:** PROD `ds4` run FROM the scallop worktree (divergent `./metal`) now
logs `Metal kernel source dir: …/ds4-metal (exe-relative)` and produces coherent
output — was garbage before. The B4 workaround is no longer needed.

## B2/B3 — --mtp-exact-sampling rewind stall (ds4.c, +86/-1) — commit 3e5c400
Under `--mtp-exact-sampling` the server resamples a tool-syntax-boundary token by
rewinding to block_start, but the only fast snapshot was taken after row 0
(snap_pos = block_start+1). rewind(block_start) never matched → `qwen4_graph_reset`
→ next eval replayed the whole kept transcript = a de-facto stall at long context
(why the flag was "never use on target machine", bug #5). Fix: add a pre-verify
snapshot (`snap0`) at block_start — lazily allocated, saved ONLY when exact
sampling is active, restored in `ds4_session_rewind` when pos == block_start.
Mirrors scallop's fix (minus its snap2/depth-2 refactor) using PROD's existing
snapshot machinery (snap0_* buffers + qwen4_graph_ensure_snap0 +
qwen4_graph_state_copy0). **Gated on exact_sampling → fully inert with the flag
off.**
**Validated:** flag-off normal MTP coherent at 43.2 t/s (unchanged); flag-on CLI
(`--mtp-exact-sampling --temp 0.7`) coherent at 40.7 t/s; and end-to-end on
ds4-server with `--mtp-exact-sampling` + tools — 4 tool-calling requests all
returned correct tool_calls in 4–5 s, server stable, zero error/stall markers
(e.g. get_weather{"city":"Paris"}, calc{"expr":"17*23"}). The stall is gone and
the flag is safe to use.

## Build / provenance
`make ds4 ds4-server ds4-bench` clean (exit 0, no errors) after each fix; the
known-good pre-fix binaries were backed up as `ds4*.baseline-6c1e836`. All fixes
are source-only (no binaries committed). The fixed binaries run the light Orca
model (and Ivan's) coherently at Ivan-parity speed.

## Preservation note
Commits live on local branch `orca-bugfixes` in the PROD (ivanfioravanti) clone,
whose remote is upstream and not ours to push. The diffs are preserved here as
patches. To publish, add the dongnh311 fork as a remote and push `orca-bugfixes`
(pending user go-ahead — an outward action).
