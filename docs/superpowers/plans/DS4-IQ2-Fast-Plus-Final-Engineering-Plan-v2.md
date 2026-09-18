# DS4-IQ2 Fast+ — Final Engineering Plan v2

---

## ✅ CONFIRMED EXECUTION PLAN (chốt 2026-09-18)

Phần này là **quyết định thực thi đã chốt với user**, ưu tiên cao hơn mọi giả định
cũ. Nó THAY THẾ thứ tự spec §25 cũ (paging/KV-first) và ground-rule "locked artifact =
OrcaUncensored".

### Mục tiêu cuối
Tạo model **DS4-IQ2 Uncensored** = **base model của Ivan** (format nhẹ: main 41.73 GiB
IQ2XXS/Q2K + PLE **Q4_1 sidecar** 29.8 GiB) **+ uncensoring từ Orca**, rồi trên model mới
đó **fix bug của Ivan + optimize theo plan v2** (execution-first). KHÔNG dùng lại format
OrcaUncensored cũ (95 GiB **BF16** n-gram embedded, gguf 147 GB — nặng gấp đôi, chậm vì
PLE I/O, phải paging).

### Quyết định codebase (quan trọng)
- **Codebase gốc = PROD** `~/.local/share/ai-gateway/ds4-metal/` (đời ~`35a5f69a`), vì nó
  hỗ trợ **PLE Q4_1 sidecar** (`--ple`) — đúng format model Ivan.
- Dev branch `dongnh311/scallop` đã **phân nhánh**: bắt buộc **BF16 n-gram embedded**, KHÔNG
  chạy được model Ivan (từ chối Q4_1). Prod thì từ chối BF16 (type 30). Hai bên không chạy
  model của nhau.
- **Hành động:** lấy prod làm gốc; **port các fix hữu ích từ scallop sang** (paged-mm
  bit-exact, MTP nextn-layer fix, memmgr adaptive cap, pager eviction leak fix, slot-sizing,
  DS4_QWEN4_MTP_STATS) — chỉ những cái áp dụng được cho path Q4_1-sidecar.

### Baseline đã đính chính (từ HF discussions của Ivan)
- **M5 Pro 64GB baseline thật = 41–45 tok/s** (MTP on + `./metal/` khớp binary). Số ~28–31
  là do **bug B4** (metal-dir mismatch → `kernel_qwen4_argmax` fail âm thầm → MTP tắt) hoặc
  đo MTP-off. Đây chính là cái bẫy các run nhanh của mình dính (chạy prod binary từ cwd
  scallop → load nhầm ./metal/ dev).
- **DoD-class ĐÃ ĐẠT trên model Ivan**: pswai đo **224K @ 45.4 tok/s decode, 323 prefill,
  50.31 GiB, ZERO swap** (M4 Max 64GB, Q2_K experts + PLE eviction + MTP). Kết luận
  "40 tok/s @220K unreachable" trước đây là do đo trên **model sai** (OrcaUncensored nặng).
- 128K OpenCode agent thực tế M5 Pro (alfredoartiles): 37–43 tok/s, ~47GB, zero crash.

### Bug list của Ivan cần fix (từ HF discussions)
- **B1 — Image cache SIGSEGV:** `server_image_cache_put` copy row 4096-wide từ embedding
  2560-wide → segfault khi request ảnh. **Fix 3 dòng đã có trong thread.**
- **B2 — `--mtp-exact-sampling` rewind stall:** ~1/10 turn stall first-token hàng phút ở
  long-ctx; draft 2-token/kept-1/resample-0 miss snapshot → `qwen4_graph_reset()` replay cả
  transcript. Refs `ds4_server.c:13670-13695`, `ds4.c:79681 ds4_session_rewind`. Fix: giữ
  snapshot valid qua end-of-reply, hoặc kept==1 & snap0_pos==pos-1 → restore snap0.
- **B3 — MTP snapshot-miss stall** (họ hàng B2, 1–8 phút, >100K ctx). Fixed trong thread #2.
- **B4 — Metal-dir mismatch → `kernel_qwen4_argmax not found` → MTP fail âm thầm → 28 t/s.**
  DS4 compile Metal kernel từ `./metal/` runtime; binary phải chạy từ dir có kernel khớp.
  → Quy tắc thao tác: **luôn chạy binary từ dir chứa ./metal/ khớp với binary đó.**

### Thứ tự thực thi đã chốt
```
0. Freeze baseline SẠCH: prod binary chạy TỪ dir của nó (./metal/ khớp) + --mtp +
   Ivan model + --ple sidecar → xác nhận output mạch lạc + ~41-45 tok/s @ short,
   và sweep ctx tới 220K/224K (đo tok/s, RAM, swap, hash). Đây là số gốc để so.
        ↓
1. Build model DS4-IQ2 Uncensored: base Ivan + uncensor Orca, GIỮ format Q4_1 PLE
   sidecar (không BF16-embed). Verify chạy được trên prod codebase + quality gate.
        ↓
2. Fix bug Ivan trên codebase gốc: B1 (segfault ảnh) → B2/B3 (MTP rewind) → B4 (deploy/
   metal-dir guard). Mỗi bug: gate + benchmark trước/sau.
        ↓
3. Optimize theo plan v2 (các phase bên dưới): audit SPECIALIZE → profile decode →
   MoE/Metal → expert cache/pager/prefetch → memmgr → MTP → Metal micro-opt →
   agent benchmark → KV cuối.
```

### Ground rules cập nhật cho hướng mới
- Artifact khoá cũ (OrcaUncensored sha ed238d8d…) **KHÔNG còn là target**; target mới là
  model DS4-IQ2 Uncensored dựng trên base Ivan. Model artifacts vẫn lên HF (repo dongnhdev),
  không commit binary/model vào git.
- SSD read-only, receipt ghi diskwrites=0 vẫn giữ.
- Gate bit-exact (resident vs paged, greedy) khi có thể; MTP lossless-greedy không cần gate.
- Không thêm user flag mới ngoài họ `--expert-*`/`--ple`/`--ssd-streaming` + env chẩn đoán.
- 1 item = 1 commit + tick progress + report.

---

## 0. Mục tiêu

Xây dựng runtime **DS4-IQ2 Fast+** cho Qwen3.8-Flash-Next trên Apple Silicon 64GB, dựa trên `ivanfioravanti/ds4-metal`.

### Baseline hiện tại

- ~49.4 tok/s decode short-context với `DS4_QWEN4_MOE_MV_SPECIALIZE=1`
- ~42.6–42.8 tok/s tại 151K–210K context
- ~430 tok/s prefill
- ~54.8GB peak wired tại 210K
- 0 real swap

### Mục tiêu

- **P0:** Không regression baseline
- **P1:** ≥50 tok/s short-context
- **P2:** ≥43 tok/s tại 150K–210K
- **P3:** Tìm đường lên 55 tok/s
- **Stretch:** 60+ tok/s nếu profiling chứng minh còn headroom

Không coi 55/60 tok/s là cam kết trước benchmark.

### Memory

- Chạy ổn trên M5 Pro 64GB
- Không phụ thuộc macOS swap
- 200K+ context ổn định
- Có headroom cho agent/tool loop
- Expert paging mở rộng working set vượt RAM
- SSD không trở thành critical-path bottleneck

### Quality

- Giữ nguyên model weights
- Không requantize trong MVP
- Greedy output deterministic
- Resident/paged path cho output/logits tương đương
- Ưu tiên bit-exact khi có thể

---

# 1. Nguyên tắc kiến trúc

## 1.1 Không rewrite DS4

Không tạo inference engine mới, không thay toàn bộ Qwen4Exp graph, không fork thành architecture khác.

```text
existing ds4-metal
        │
        ├── existing Qwen4Exp graph
        ├── existing Metal kernels
        ├── existing Fast/resident path
        │
        └── Fast+ additions
              ├── execution optimizations
              ├── expert cache
              ├── expert pager
              ├── async prefetch
              └── memory manager
```

## 1.2 Fast/resident path là canonical path

Nếu expert nằm trong RAM:

```text
router
  ↓
expert cache HIT
  ↓
existing Metal expert path
```

Phải nhanh tương đương hoặc nhanh hơn DS4 hiện tại.

Không biến toàn bộ inference thành SSD-streaming inference.

## 1.3 SSD là backing store, không phải accelerator

Không mặc định SSD paging sẽ tăng tok/s.

SSD dùng để:

1. giảm RAM pressure
2. cho working set lớn hơn RAM
3. giữ cold experts ngoài RAM
4. prefetch trước khi cần
5. mở đường cho model/context lớn hơn

Nếu SSD không overlap hoàn toàn với compute thì phải đo rõ chi phí.

---

# 2. Phase 0 — Freeze baseline

Benchmark:

```text
8K
32K
64K
128K
151K
180K
210K
220K
256K nếu runtime chịu được
```

Mỗi benchmark ghi:

```text
TTFT
prefill tok/s
decode tok/s
E2E tok/s
tokens generated
RAM
wired memory
swap
SSD read/write
Metal execution time
CPU time
expert dispatch time
synchronization time
MTP acceptance
output hash
```

Chạy tối thiểu:

```text
MTP OFF
MTP ON
SPECIALIZE OFF
SPECIALIZE ON
```

Baseline phải lưu commit + command + environment.

---

# 3. Phase 1 — Mổ `SPECIALIZE=1`

Đây là ưu tiên số 1.

Commit:

```text
35a5f69a
```

Benchmark:

```text
46.4 → 49.4 tok/s
≈ +6.2–6.45%
bit-exact
```

Phải xác định chính xác gain đến từ đâu.

Audit:

```text
ds4.c
Qwen4Exp graph
MoE dispatch
expert tensor loading
Metal kernel selection
Metal specialization
buffer layout
tensor dimensions
threadgroup sizing
SIMD grouping
kernel launch count
synchronization
temporary buffers
memory copies
branching
```

Tìm toàn bộ call chain của:

```text
DS4_QWEN4_MOE_MV_SPECIALIZE
```

Phải trả lời source-level:

> SPECIALIZE=1 thực sự tối ưu cái gì?

Không chấp nhận giải thích chung chung.

---

# 4. Phase 2 — Profile decode execution

Phân rã một token:

```text
token
 │
 ├── embedding
 ├── GDN
 ├── QSA
 ├── router
 ├── expert selection
 ├── expert gate/up
 ├── expert down
 ├── aggregation
 ├── hyperconnection
 ├── output
 └── MTP
```

Đo:

```text
CPU dispatch
Metal launch
GPU kernel
memory bandwidth
synchronization
expert dispatch
expert GEMM
GDN
QSA
MTP
```

Tạo breakdown thời gian:

```text
decode token =

X ms GDN
Y ms QSA
Z ms MoE
A ms dispatch
B ms synchronization
C ms other
```

Chỉ sau đó mới quyết định kernel nào đáng tối ưu.

---

# 5. Phase 3 — Optimize MoE execution

## 5.1 Expert dispatch

Kiểm tra:

- router output → expert ID conversion
- duplicate expert handling
- token-to-expert mapping
- gather/scatter
- expert ordering
- launch count
- buffer allocation
- CPU↔GPU synchronization

Mục tiêu:

```text
router
  ↓
compact expert IDs
  ↓
single efficient dispatch
  ↓
Metal
```

Giảm CPU work, kernel launches, synchronization và memory copies.

## 5.2 Expert tensor layout

DS4-IQ2:

```text
gate/up = IQ2_XXS
down    = Q2_K
```

Không requantize trong phase này.

Kiểm tra:

- alignment
- block layout
- contiguous access
- decode-specific kernels
- shared memory
- SIMD utilization
- dequant overhead
- expert fusion

Mục tiêu: cùng tensor, cùng output, ít overhead hơn.

---

# 6. Phase 4 — Expert bundle format

Mỗi expert là một logical bundle:

```text
Expert(layer, id)
 ├── gate IQ2
 ├── up IQ2
 └── down Q2K
```

Đề xuất:

```text
qwen38-experts.bin
```

Không dùng hàng nghìn file nhỏ.

Index:

```text
layer_id
expert_id

gate_offset
gate_size

up_offset
up_size

down_offset
down_size
```

Hoặc contiguous bundle:

```text
[layer][expert]
    gate
    up
    down
```

---

# 7. Phase 5 — Resident expert cache

Implement:

```text
ds4_expert_cache.c
ds4_expert_cache.h
```

Logic:

```text
lookup(layer, expert)
    ↓
HIT  → existing Metal path
MISS → pager
```

Cache key:

```text
(layer_id, expert_id)
```

Policy ban đầu: **LRU**.

Sau đó có thể thử frequency + recency.

Configurable:

```text
--expert-cache
--expert-cache-size
```

Không hard-code một mức RAM.

---

# 8. Phase 6 — Expert pager

Implement:

```text
ds4_expert_pager.c
ds4_expert_pager.h
```

Flow:

```text
router
   ↓
selected experts
   ↓
cache lookup
   ├── HIT
   │    ↓
   │  Metal
   │
   └── MISS
        ↓
      pager
        ↓
      SSD
        ↓
      staging buffer
        ↓
      cache
        ↓
      Metal
```

Pager dùng:

- contiguous file
- `pread`/equivalent
- aligned reads
- reusable staging buffers
- bounded queue

Tránh file-open/read/close cho từng tensor.

---

# 9. Phase 7 — Async SSD prefetch

Implement:

```text
ds4_expert_prefetch.c
ds4_expert_prefetch.h
```

Nguyên tắc:

```text
Layer N compute
       │
       ├── current experts
       │
       └── prefetch Layer N+1
                     │
                     ▼
                   SSD
                     │
                     ▼
                    RAM
```

Tránh:

```text
need expert
   ↓
synchronous SSD read
   ↓
wait
   ↓
compute
```

Config:

```text
--expert-prefetch
--expert-prefetch-depth
```

---

# 10. Phase 8 — Predictive prefetch

Sau khi basic pager ổn.

Dùng selected experts hiện tại để chuẩn bị layer tiếp theo.

Score ban đầu:

```text
score =
    frequency
  + recency
  + next-layer probability
  - load cost
```

Không over-engineer ML predictor.

Đo:

```text
prefetch hit rate
SSD latency
overlap %
stall time
```

---

# 11. Phase 9 — Unified memory manager

Quản lý:

```text
model
KV
expert cache
Metal workspace
staging buffers
MTP
```

Implement:

```text
ds4_memory_manager.c
ds4_memory_manager.h
```

Theo dõi:

```text
resident model
expert cache
KV/cache state
Metal workspace
staging
OS headroom
```

Có:

```text
hard budget
soft budget
eviction
pressure handling
```

Không để expert cache đẩy macOS vào swap.

---

# 12. Phase 10 — MTP optimization

Chỉ tối ưu MTP sau normal decode path.

Đo:

```text
MTP OFF
MTP ON
```

Phải đo:

```text
accepted tokens
rejected tokens
verification cost
draft cost
net tok/s
```

Không giả định `--mtp-draft 7` tạo chain 7 draft tokens trong Qwen3.8.

Tôn trọng graph implementation thực tế.

---

# 13. Phase 11 — KV research, KHÔNG phải MVP

Benchmark hiện tại:

```text
151K → 210K

42.8 → 42.6 tok/s

54.7GB → 54.8GB wired
```

Qwen3.8 Flash-Next có compressed/linear-attention behavior.

Do đó chưa ưu tiên:

```text
FP8 KV
paged KV
KV quantization
KV eviction
aggressive KV compression
```

Chỉ quay lại khi profiler chứng minh KV là bottleneck.

Nếu cần nghiên cứu, bắt đầu bằng paged/block KV abstraction, chưa thay format KV ngay.

---

# 14. Phase 12 — Metal micro-optimization

Sau profiling.

### Kernel specialization

Kiểm tra compile-time specialization cho các dimension cố định.

### Fusion

Chỉ fuse khi launch overhead và memory traffic giảm mà occupancy không xấu đi.

### Buffer reuse

Giảm:

```text
allocate
free
copy
synchronize
```

trong mỗi token.

### Decode-specific path

Prefill và decode có workload khác nhau; không nhất thiết dùng cùng kernel strategy.

---

# 15. Phase 13 — Benchmark matrix

Mỗi optimization chạy:

```text
8K
32K
64K
128K
151K
180K
210K
220K
256K+
```

Đo:

```text
prefill
decode
E2E
TTFT
RAM
wired
swap
SSD bandwidth
cache hit rate
prefetch hit rate
stall %
```

và:

```text
MTP OFF
MTP ON
```

Nếu thay đổi execution:

```text
SPECIALIZE OFF
SPECIALIZE ON
```

---

# 16. Correctness gate

## Output

```text
same prompt
same seed
same sampling
same greedy output
```

## Logits

So sánh:

```text
max abs delta
mean abs delta
top-k agreement
```

## Resident vs paged

Test:

```text
all resident
partial resident
full paging
mixed
```

Expected:

```text
same output
```

Ưu tiên bit-exact nếu floating-point execution cho phép.

---

# 17. Performance regression gate

Không merge nếu:

```text
short decode regression > threshold
```

hoặc:

```text
210K regression
```

trừ khi optimization có mục tiêu memory/scale rõ ràng và regression được chấp nhận có chủ ý.

Mỗi commit ghi:

```text
before
after
delta
workload
hardware
runtime flags
```

---

# 18. Runtime flags

Default behavior phải giữ tương thích DS4 hiện tại.

```text
--expert-cache
--expert-cache-size <MB>
--expert-prefetch
--expert-prefetch-depth <N>
--expert-cache-policy <lru|freq>
--memory-budget <GB>
```

Environment/debug:

```text
DS4_QWEN4_MOE_MV_SPECIALIZE=1
DS4_EXPERT_CACHE_DEBUG=1
DS4_EXPERT_PAGER_DEBUG=1
DS4_EXPERT_PREFETCH_DEBUG=1
```

---

# 19. Telemetry / receipts

Runtime phải có:

```text
expert_cache_hits
expert_cache_misses

expert_prefetch_requests
expert_prefetch_hits
expert_prefetch_misses

ssd_read_bytes
ssd_read_time

expert_load_time
expert_stall_time

cache_evictions

resident_experts
paged_experts
```

Cho mỗi generation/turn nếu có thể.

---

# 20. Agent benchmark

Test thực tế:

```text
Claude Code
Codex
OpenCode
```

Workload:

```text
repo exploration
search
edit
compile
test
bugfix
refactor
multi-turn
tool loop
```

Đo:

```text
time/turn
time-to-first-useful-token
decode throughput
tool latency
context size
cache reuse
```

Mục tiêu cuối:

> faster useful agent completion

không phải chỉ một con số tok/s đẹp.

---

# 21. Definition of Done

## Correctness

- Existing Qwen3.8 DS4 output preserved
- Resident path pass
- Paged path pass
- Deterministic greedy pass
- No unexplained logits divergence

## Performance

- No regression từ current ~49.4 tok/s baseline
- Long-context giữ khoảng ≥42 tok/s ở 150–210K
- Có measured improvement từ execution optimization
- Mọi speedup có benchmark evidence

## Memory

- 64GB ổn định
- 200K+ context
- Không uncontrolled swap
- Expert cache có budget
- SSD paging không phá runtime

## Paging

- Single contiguous expert store
- Layer/expert index
- RAM cache
- Async SSD loading
- Prefetch
- Observable hit/miss/stall metrics

## Architecture

- Existing DS4 Fast path retained
- No full graph rewrite
- No new inference engine
- No mandatory SSD per-token path
- Features independently switchable

---

# 22. Thứ tự triển khai chính thức

Không đảo thứ tự nếu chưa có profiling evidence:

```text
0. Freeze baseline
        ↓
1. Audit 35a5f69a / SPECIALIZE=1
        ↓
2. Profile complete decode path
        ↓
3. Optimize MoE dispatch / Metal execution
        ↓
4. Re-benchmark
        ↓
5. Build expert bundle/index
        ↓
6. Resident expert cache
        ↓
7. Async SSD pager
        ↓
8. Predictive prefetch
        ↓
9. Unified memory manager
        ↓
10. MTP optimization
        ↓
11. Metal micro-optimization
        ↓
12. Agent benchmark
        ↓
13. KV research ONLY if profiling demands it
```

---

# 23. Strategic decision

Không còn coi:

```text
KV growth
```

là bottleneck mặc định.

Không coi:

```text
SSD
```

là speed accelerator mặc định.

Không coi:

```text
model size > RAM
```

đồng nghĩa với paging phải là optimization đầu tiên.

Thay vào đó:

> **Tối ưu execution path trước, vì benchmark đã chứng minh execution specialization có thể tăng tốc bit-exact. Sau đó xây expert cache/pager như memory hierarchy để mở rộng khả năng chạy trên 64GB.**

Đây là nền tảng của **DS4-IQ2 Fast+**.

---

# 24. Success hypothesis

```text
Current
49.4 short
42.6 @ 210K
        │
        ▼
SPECIALIZE audit
        │
        ▼
MoE/Metal optimization
        │
        ├── +?
        ▼
50–55 tok/s region
        │
        ▼
expert cache/prefetch
        │
        ├── memory headroom
        ├── larger working set
        └── potentially hidden SSD latency
        │
        ▼
MTP optimization
        │
        ▼
55+ / potentially 60+
```

**55–60 tok/s là target nghiên cứu, không phải cam kết.**
