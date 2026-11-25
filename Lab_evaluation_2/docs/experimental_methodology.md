# Experimental Methodology & Benchmarking Protocols

## 1. Benchmarking Objectives

The primary objective of this benchmarking suite is to empirically analyze:
1. Contention bottlenecks in GPU atomic updates across global memory vs on-chip shared memory.
2. The sensitivity of histogram execution time to input data distributions (Uniform, Gaussian, Highly Skewed).
3. The impact of bin-count granularity ($B \in \{16, 64, 256\}$) on shared-memory initialization and merge overhead.
4. Large-scale performance behavior on real-world high-resolution image datasets (DIV2K, 6.69 billion values).

---

## 2. Timing Methodology & Measurement Boundaries

To ensure rigorous and reproducible results, timing measurements are partitioned into two strictly separated metrics:

### A. GPU Kernel Execution Time
- Captured using high-resolution hardware timers via CUDA Events:
  ```cuda
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  kernel<<<grid, block, shared_bytes>>>(...);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  ```
- Excludes host-to-device memory allocation, PCIe transfers, and disk I/O.
- Each kernel benchmark was executed with **10 GPU warm-up and timing repetitions**; mean times are reported.

### B. End-to-End Wall-Clock Time
- Captured on host CPU via `std::chrono::high_resolution_clock` or `time.perf_counter()`.
- Encompasses:
  1. Reading chunk data from NVMe disk storage.
  2. Host-to-Device (H2D) PCIe Gen4 memory copy.
  3. Kernel execution and synchronization.
  4. Device-to-Host (D2H) transfer of partial 256-bin histograms.
  5. 64-bit host counter aggregation.

---

## 3. Streaming Model for Large Datasets (DIV2K)

Processing 6,691,265,280 elements (~6.23 GiB) at once can lead to device allocation failure on limited VRAM configurations. Therefore:
- The input binary is divided into contiguous **64 MiB streaming chunks** ($67,108,864$ elements per chunk).
- A reusable pinned or device buffer of 64 MiB is allocated once.
- Intermediate GPU histograms use 32-bit unsigned integer counters (`uint32_t`).
- Host aggregation uses **64-bit unsigned integers (`uint64_t`)** to ensure no overflow occurs when summing counts up to $6.69 \times 10^9$.

---

## 4. Correctness Verification Protocol

Every benchmark execution mandates automatic verification against the single-core CPU reference:
1. **Total Element Count Test:**
   $$\sum_{b=0}^{B-1} \text{hist}[b] == N$$
2. **Bin-by-Bin Identity Test:**
   $$\text{hist}_{\text{GPU}}[b] == \text{hist}_{\text{CPU}}[b] \quad \forall b \in [0, B-1]$$
3. An execution is marked **PASSED** only if both assertions hold true with zero discrepancy.
