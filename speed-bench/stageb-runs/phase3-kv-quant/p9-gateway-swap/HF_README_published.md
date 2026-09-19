---
license: apache-2.0
base_model:
  - orcarouter/Qwen3.8-Flash-Next-Uncensored
  - ivanfioravanti/Qwen3.8-Flash-Next-DS4-IQ2
tags:
  - qwen4exp
  - ds4-metal
  - iq2
  - uncensored
  - mtp
  - q4k
---

# Qwen3.8-Flash-Next OrcaUncensored — DS4 IQ2 (Light)

Qwen3.8-Flash-Next in the **light DS4-IQ2 packaging** of
[ivanfioravanti/Qwen3.8-Flash-Next-DS4-IQ2](https://huggingface.co/ivanfioravanti/Qwen3.8-Flash-Next-DS4-IQ2):
an IQ2 main (IQ2_XXS gate/up, Q2_K down padded to 768, embedded MTP block) **plus an
external demand-paged PLE Q4_1 sidecar** instead of a ~95 GiB resident BF16 n-gram.
Runs resident and **zero-swap on a 64 GiB Apple Silicon box** (measured on M5 Pro),
8K→220K context.

## Performance (M5 Pro 64 GiB, measured)

| model | MTP-off | single-stream `--mtp` | notes |
|---|---|---|---|
| Q8 dense (exact) | ~30–32 t/s | ~38 t/s | most accurate |
| **Q4_K imat dense** | ~36 t/s | **~42–43 t/s** | **+~14% vs Q8**, near-Q8 quality |

- **MTP** (`--mtp`) adds ~+17% single-stream over MTP-off; draft acceptance ~67%.
- **Adaptive draft depth** (engine env `DS4_QWEN4_MTP_DEPTH`, default auto): drafts a
  2nd token on deterministic/structured/code continuations for a further **+5–6%**,
  and falls back on free-form prose so it never regresses. Output stays
  autoregressive-exact (verify only commits argmax-matching drafts → zero quality
  change).
- **Multi-session throughput** (engine `--batched-session N`, concurrent requests,
  full quality): **~67 t/s @ 8 streams, ~80 t/s @ 16 streams aggregate** (~2× a
  single stream). Decode is memory-bandwidth-bound on this hardware, so aggregate
  throughput plateaus around ~80 t/s rather than scaling linearly; a single stream
  cannot exceed ~43 t/s here (that ceiling needs a smaller model or more MTP heads,
  not tuning). Concurrency is bounded by ctx × sessions KV vs 64 GiB (ctx 4096 fits
  16 streams; large ctx needs fewer).

## Two dense-quant mains (pick one; both share the same PLE sidecar)

The IQ2 experts are identical across both; they differ only in how the per-layer
**dense** projections are quantized:

| main | dense | size | decode vs Q8 | logit cosine vs Q8 | pick when |
|---|---|---|---|---|---|
| `...-Q2KDownPad768-MTP.gguf` | **Q8_0** | 41.73 GiB | baseline | 1.000 (exact) | most accurate output |
| `...-DenseQ4Kselimat-MTP.gguf` | **selective Q4_K + imatrix** | 40.68 GiB | **+~14%** | **0.949** (argmax preserved) | faster decode, near-Q8 quality |

The **DenseQ4Kselimat** variant requantizes the less-sensitive dense projections
Q8_0→Q4_K (routed to DS4's fast dense Q4_K GEMV) while keeping the most
quality-sensitive full-attention **q/k/v/output** projections at Q8_0, and weights
each dense tensor's Q4_K quantization by an **importance matrix** (activation energy
per input column). Net: ~+14% decode at logit cosine ~0.949 vs Q8 (argmax preserved,
coherent). Encode-only change (same runtime kernel/bytes as plain Q4_K; the imatrix
costs nothing at inference). imatrix source: the published llama.cpp calibration
`unsloth/Qwen3.8-Flash-Next-GGUF` (same qwen4exp tensor decomposition).

## Files
- `Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q2KDownPad768-MTP.gguf` — **Q8 dense main**
  (exact). 44,806,612,448 bytes, sha256
  `e078c60abfdc9c5dd849eddb41660e5fc8f1b0da2f1200c643a8d3ec50324e8a`.
- `Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q2KDownPad768-DenseQ4Kselimat-MTP.gguf` —
  **imatrix-weighted selective-Q4K dense main** (faster, near-Q8). 43,677,591,168
  bytes, sha256 `b1dd08509231126b5f1596603fd2589ff8bdf8eae221420f3e4409110fee8bd4`.
- `Qwen3.8-Flash-Next-PLE-Q4_1.gguf` — PLE Q4_1 n-gram sidecar, **shared by both
  mains** (required). Byte-identical between base Qwen and this fine-tune
  (abliteration does not touch the n-gram table), so it is Ivan's sidecar verbatim.

## Runtime (DS4 Metal engine)
Runs on the **ds4-metal** engine (llama.cpp-lineage Metal fork for qwen4exp). The
build used here adds, over the stock light runtime: the external `--ple` sidecar
loader combined with native **multi-session batched decode**, the fast **Q4_K dense
GEMV**, **adaptive MTP draft depth**, and correctness fixes (accurate GDN-gate
softplus, an isfinite guard in the expert reduction, `ignore_eos` honored under
`--mtp`, and a clearer tool-call parse error). The `--ple` sidecar is CPU-mmap
demand-paged (each token faults ~one page per hash head); it fits 64 GiB beside the
resident model, whereas the embedded-BF16 n-gram lineage does not.

## Usage
```
# exact (default)
ds4 -m Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q2KDownPad768-MTP.gguf \
    --ple Qwen3.8-Flash-Next-PLE-Q4_1.gguf --mtp -c 4096 -p "..."

# faster (imatrix-weighted selective dense Q4_K)
ds4 -m Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q2KDownPad768-DenseQ4Kselimat-MTP.gguf \
    --ple Qwen3.8-Flash-Next-PLE-Q4_1.gguf --mtp -c 4096 -p "..."

# serving many concurrent requests (ds4-server, ~2x aggregate)
ds4-server -m ...DenseQ4Kselimat-MTP.gguf --ple ...PLE-Q4_1.gguf --mtp \
    --batched-session 16 -c 4096 --host 127.0.0.1 --port 8000
```
Long contexts: add `--prefill-chunk 2048` to stay zero-swap at 128K–220K on 64 GiB.

## Lineage / method
Uncensored weights from `orcarouter/Qwen3.8-Flash-Next-Uncensored`
(rev `8336e613ea508b13c2159bd0f68965d97a606b95`), quantized to Ivan's DS4-IQ2 layout,
then repackaged into this light form by stripping the embedded n-gram tensor
(retained tensors byte-identical; external Q4_1 sidecar substituted). The
DenseQ4Kselimat main is derived from the Q8 main by imatrix-weighted Q4_K requant of
the dense projections (attn q/k/v/output kept Q8_0).

## Behavior — uncensoring is PARTIAL (honest note)
This is an abliterated fine-tune, but the **light IQ2 quantization dilutes most of
the abliteration**: after quantizing to 2-bit experts + Q8 dense, this model differs
from the equivalent base build in only a small set of dense projections (the
GDN linear-attention output weights), because a 2-bit round-trip erases the small
weight edits abliteration makes elsewhere. As a result the model's refusal behavior
is **much closer to the base model than to a full abliteration**: it relaxes some
mild refusals but **still refuses most harmful requests, and all extreme categories
(e.g. weapon/drug synthesis)**. Treat it as a *lightly* uncensored research artifact,
not a fully unrestricted model. Provided for research; you are responsible for lawful
use.
