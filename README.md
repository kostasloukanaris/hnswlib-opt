# hnswlib-opt

> An optimized fork of [nmslib/hnswlib](https://github.com/nmslib/hnswlib) focusing on cache-aware memory layout, explicit AVX-512 SIMD vectorization, and multi-threaded query processing on AMD Zen 5.

This repository is a **personal learning and practice project** in High-Performance Computing, undertaken as a capstone during the final semesters of a Computer Engineering and Informatics degree. The goal is to apply industry-standard optimization techniques to a real, production-used open-source library and document every decision with hardware counter evidence.

All modifications to original source files are documented in the commit history with explanations of what was changed and why.

---

## Original Library

This is a fork of **[nmslib/hnswlib](https://github.com/nmslib/hnswlib)** by Yu. A. Malkov and D. A. Yashunin.
Reference paper: [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320) (2018).

---

## Hardware Target

| Component | Details |
|---|---|
| CPU | AMD Ryzen 7 9700X (Zen 5, 8 cores, no hyperthreading) |
| SIMD | AVX2 + AVX-512 (avx512f, avx512dq, avx512bw, avx512vl) |
| L1 / L2 / L3 | 48 KB / 1 MB / 32 MB |
| Memory | DDR5 ~80 GB/s |
| GPU | AMD RX 6800 XT (RDNA 2, 16 GB GDDR6) |
| OS | Kubuntu Linux |

---

## Benchmark

The benchmark harness (`bench.cpp`) is written from scratch. It loads the SIFT1M dataset, builds the HNSW index, runs 10,000 queries, and reports QPS and Recall@10.

### Dataset

```bash
mkdir datasets && cd datasets
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzvf sift.tar.gz
```

### Build and Run

```bash
cd build
cmake .. && make bench_release -j8
cd ..

./build/bench_release

```

---

## Optimization Log

### Phase 0 — Baseline (Frozen Reference)

> No source modifications. This is the reference point all future optimizations are measured against.

```
Date:     2026-03-07
Machine:  AMD Ryzen 7 9700X
Build:    g++ -O3 -march=native -mavx2 -mavx512f -mavx512dq -mfma
Dataset:  SIFT1M (1,000,000 vectors, 128-dim float32)
Index:    M=16, ef_construction=200, ef_search=50, k=10

Run 1:  QPS = 17631.9   Recall@10 = 0.9465   Time = 0.567s
Run 2:  QPS = 17747.6   Recall@10 = 0.9465   Time = 0.563s
Run 3:  QPS = 17578.7   Recall@10 = 0.9465   Time = 0.569s
Run 4:  QPS = 17752.4   Recall@10 = 0.9465   Time = 0.563s

Mean QPS:     17677.7
Min  QPS:     17578.7
Max  QPS:     17752.4
Variance:     0.98%     PASS (threshold: <2%)
Recall@10:    0.9465    stable across all runs
```

Initial `perf stat` findings:

```
IPC:               0.88     (below 1.0 — memory-bound, CPU is starving for data)
Branch-miss rate:  4.56%    (4.5B mispredictions — unpredictable conditionals in search loop)
Frontend stalls:   11.30%   (pipeline starved, partly from branch misses)

Cache-references:  183.2B
Cache-misses:       35.2B   (19.23% of all cache refs — nearly 1 in 5 misses)
dTLB-loads:          9.3B
dTLB-load-misses:    3.6B   (38.83% of dTLB accesses — high TLB pressure from pointer-chasing)
```

**Conclusion:** The program is memory-bound, not compute-bound. The CPU is capable of far
more — it is waiting for data. The graph traversal pattern (pointer-chasing across scattered
heap allocations) is the primary cause. This informs the focus of the upcoming phases.

---

### Phase 1 — Profiling and AVX-512 fmadd Optimization

#### Profiling

Three tools were used to locate the bottleneck:

**`perf stat` (query path only)**

```
IPC:              1.19    (CPU stalling on memory — below compute saturation)
L1-miss rate:     6.79%
LLC-miss rate:    19.82%  (nearly 1 in 5 cache accesses goes to RAM)
Branch-miss rate: 1.57%   (priority queue heap comparisons unpredictable)
```

**`perf record -g`** (call-graph sampling)

`L2SqrSIMD16ExtAVX512` accounted for **47.55% self-time** — the single hottest function. The profile was dominated by the index build (180 s vs 0.56 s for queries), which is why `--query` mode was added to `bench.cpp` for clean query-path measurements.

**Assembly inspection + vectorization report**

`objdump` confirmed AVX-512 is active (`zmm` registers). The GCC vectorization report (`-fopt-info-vec`) revealed that the AVX-512 hot loop was emitting two separate instructions (`vmulps` + `vaddps`) where a single fused multiply-add was available.

#### Optimization: fmadd in `L2SqrSIMD16ExtAVX512`

`hnswlib/space_l2.h` — uncommented the existing `_mm512_fmadd_ps` call:

```cpp
// Before: 2 instructions
sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));

// After: 1 instruction
sum = _mm512_fmadd_ps(diff, diff, sum);
```

`_mm512_fmadd_ps` computes `diff * diff + sum` as a single fused operation — lower latency and higher throughput on Zen 5's FMA units.

#### Results

| Metric | Baseline | Phase 1 | Delta |
|---|---|---|---|
| QPS | 17,677.7 | **18,547.9** | **+4.9%** |
| Recall@10 | 0.9465 | 0.9465 | 0 |
| IPC | 1.19 | 1.24 | +4.2% |
| LLC-miss rate | 19.82% | 18.99% | −0.4pp |
| Branch-miss rate | 1.57% | 1.56% | ~0 |

IPC increase confirms fewer instructions per loop iteration. Memory and branch counters are flat — as expected for a compute-side change with no effect on access patterns.

---

### Phase 2 — Memory Layout Optimizations

Two structural memory problems were identified from Phase 1 profiling and addressed together.

#### Problem 1: Scattered upper-layer neighbor lists

The original code called `malloc` once per node during `addPoint`, scattering ~62,500 independent allocations randomly across the heap. Every `get_linklist` call during graph traversal dereferenced a pointer to a random address, causing TLB misses and poor cache locality.

**Fix — arena allocator:** a single contiguous block is pre-allocated upfront. Every node's upper-layer neighbor list lives at a fixed, predictable offset:

```cpp
linklists_arena_ = (char *) malloc(max_elements_ * linklists_slot_size_);
linkLists_[i] = linklists_arena_ + i * linklists_slot_size_;  // node i's slot
```

This eliminates ~62,500 `malloc`/`free` calls during build and consolidates all neighbor data into one memory region. The kernel automatically backed this large contiguous allocation with 2 MB huge pages, collapsing dTLB misses from **3.65 billion → ~13 million** (270×).

#### Problem 2: Misaligned vector data

Each node's 128-float vector started at byte offset 132 within its slot. Since 132 % 64 = 4, every AVX-512 load of 16 floats (64 bytes) crossed a cache line boundary, requiring two cache line fetches where one should suffice.

**Fix — vector-first layout + `aligned_alloc`:** the slot was reordered so the vector occupies offset 0, and the slot size padded to a multiple of 64:

```
Before: [ neighbor list: 132 B | vector: 512 B | label: 8 B ]  = 652 B (misaligned)
After:  [ vector: 512 B | neighbor list: 132 B | label: 8 B | padding: 52 B ] = 704 B (aligned)
```

With `aligned_alloc(64, ...)` for the base pointer and a slot size of 704 (= 11 × 64), every node's vector is guaranteed 64-byte aligned — eliminating all cache line splits in the AVX-512 distance loop.

`realloc` does not preserve alignment, so `resizeIndex` was updated to use `aligned_alloc + memcpy + free`.

#### Results

| Metric | Baseline | Phase 1 | Phase 2 | Delta vs baseline |
|---|---|---|---|---|
| QPS | 17,677.7 | 18,547.9 | **19,624.8** | **+11.0%** |
| Recall@10 | 0.9465 | 0.9465 | 0.9465 | 0 |
| IPC | 1.19 | 1.24 | 1.32 | +10.9% |
| L1-miss rate | 6.79% | 6.80% | 6.58% | −0.2pp |
| Branch-miss rate | 1.57% | 1.56% | 1.42% | −0.1pp |
| dTLB-load-misses | ~3.65 B | — | ~13 M | −270× |

LLC-miss *rate* rose from 18.99% to 22.43% despite absolute misses being flat — total LLC accesses dropped 14% (fewer wasted cache-line fetches per distance call), which raises the ratio while the underlying miss count is unchanged.

---

## License

Apache License 2.0 — same as the original [nmslib/hnswlib](https://github.com/nmslib/hnswlib).
Original copyright notices and the `LICENSE` file are preserved intact.
Modified source files carry notices of modification in the commit history.

---

*Konstantinos Loukanaris — 2026*