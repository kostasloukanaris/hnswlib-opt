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

## License

Apache License 2.0 — same as the original [nmslib/hnswlib](https://github.com/nmslib/hnswlib).
Original copyright notices and the `LICENSE` file are preserved intact.
Modified source files carry notices of modification in the commit history.

---

*Konstantinos Loukanaris — 2026*