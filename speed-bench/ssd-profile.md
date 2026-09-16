# SSD Profile for M5 Pro 64GB

## System
- **Machine**: Apple M5 Pro (Mac17,8)
- **Storage**: Internal APFS, 926 GiB volume
- **Free space**: ~327 GiB
- **macOS**: 26.6.2 (25G83)

## Read Performance

### Sequential Read (1MB blocks)
- **Speed**: ~14.1 GB/s (14107 MB/s)
- **Method**: `dd if=model.gguf of=/dev/null bs=1M count=1024 iflag=direct`

### 4K Random Read (simulating expert bundle preads)
- **Speed**: ~72.9 MB/s
- **Method**: `dd if=model.gguf of=/dev/null bs=4k count=1000 iflag=direct`

## Implications for Expert Cache

With qwen38-experts.bin ≈ 36-40 GiB:
- **Sequential load time**: ~2.5-2.8 seconds (best case)
- **Random read time** (for scattered expert bundles): significantly higher
- **Prefetch budget**: p50 latency at ~4K random ≈ 0.5μs per 4K block

**Recommendation**: Bundle experts in contiguous layout to maximize sequential reads; batch preads where possible.

