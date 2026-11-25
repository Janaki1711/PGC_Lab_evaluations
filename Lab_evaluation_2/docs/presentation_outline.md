# Academic Presentation Outline: Lab Evaluation 2

**Theme 3: Histogram Computation with Atomics and Privatization**  
**Course:** Parallel & GPU Computing (CS60012)  
**Presenter:** Janaki M  
**System Hardware:** NVIDIA GeForce RTX 5060 Ti (16 GB, sm_120) | Intel Core i5-13400 (32 GB RAM)  

---

## Checkpoint 1: Problem Statement & Motivation
- **Definition of Histogramming:** Frequency tallying over discrete intensity bins $[0, 255]$ for 8-bit image elements.
- **The Core Parallel Bottleneck:** Atomic contention. In parallel SIMT execution, thousands of threads competing to update the same memory address cause serialization at memory controllers.
- **Objective:** Quantify the performance gap between Global Atomics and Shared-Memory Privatization across synthetic distributions, varying bin counts, and real-world 6.69B DIV2K image data.

---

## Checkpoint 2: Architectural Models & Implementations
- **CPU Baseline:** Single-threaded scalar accumulation in host memory.
- **Model 1 — CUDA Global Atomics:**
  - Threads execute grid-stride loops and issue `atomicAdd()` directly to global VRAM.
  - Vulnerability: Catastrophic serialization when input values concentrate into a small subset of bins.
- **Model 2 — CUDA Shared-Memory Privatization:**
  - Block-private sub-histograms in fast on-chip SRAM (`__shared__`).
  - Local low-latency atomic accumulation.
  - Single cooperative block-merge reduction to global memory per bin.

---

## Checkpoint 3: Experimental Configuration & Benchmarking Suite
- **Hardware Platform:** NVIDIA GeForce RTX 5060 Ti (Compute Capability 12.0, CUDA 13.5, MSVC 2022).
- **Synthetic Benchmarks:** $2^{24} = 16,777,216$ elements under Uniform, Gaussian, and 90% Skewed distributions.
- **Bin-Count Sensitivity:** Scaling across $B \in \{16, 64, 256\}$ bins.
- **Real-World High-Resolution Dataset:** DIV2K Training Set (800 images, 2.23B pixels, 6.69B raw RGB bytes).

---

## Checkpoint 4: Empirical Results & Performance Analysis
- **Synthetic Contention Results:**
  - Uniform: Global $2.43\text{ ms}$ vs Shared $0.052\text{ ms}$ ($46.96\times$ speedup ratio).
  - Highly Skewed (90% Mode): Global jumps to $6.40\text{ ms}$, while Shared remains at $0.051\text{ ms}$ (**126.85× improvement**).
- **Bin-Count Trade-off:** Shared-memory kernel duration scales from $0.027\text{ ms}$ (16 bins) to $0.048\text{ ms}$ (256 bins) due to block-merge overhead ($65,536$ vs $1,048,576$ updates).
- **DIV2K Large-Scale Benchmark:**
  - CPU Baseline: $5853.40\text{ ms}$
  - Global Atomic Kernel: $736.96\text{ ms}$ ($7.94\times$ speedup)
  - Shared-Memory Kernel: $50.64\text{ ms}$ (**115.60× speedup vs CPU, 14.55× vs Global**)
  - Wall-clock time dominated by PCIe Gen4 H2D transfer ($977\text{ ms}$) and streaming I/O.
- **Verification:** 100% exact numerical match across all executions.

---

## Checkpoint 5: Conclusions & AI Applications
- **AI Application 1 (Quantization Calibration):** Percentile clipping ($99.99\%$) reduces quantization MSE by $23\times$ on Gaussian activation tensors.
- **AI Application 2 (GPU Parallel CDF):** Hillis-Steele inclusive prefix sum computes cumulative distribution on-chip in $0.0069\text{ ms}$.
- **Final Summary:** Shared-memory privatization eliminates atomic contention bottlenecks and is essential for high-throughput image and tensor processing pipelines.
