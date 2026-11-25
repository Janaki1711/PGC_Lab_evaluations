# Technical Report: CUDA-Accelerated Histogram Computation

**Project Title:** CUDA-Accelerated Histogram Computation Using Global Atomics and Shared-Memory Privatization on Large-Scale Image Datasets  
**Academic Theme:** Theme 3 – Histogram with Atomics and Privatization  
**Author:** Janaki M  
**Repository:** [PGC_Lab_evaluations](https://github.com/Janaki1711/PGC_Lab_evaluations)  

---

## 1. Abstract & Problem Statement

Histogram computation is an essential algorithmic primitive in digital image processing, statistical data analysis, and deep neural network quantization calibration. Given an input dataset of discrete elements, a histogram tallies the frequency of occurrence of values across defined bins. For standard 8-bit unsigned integer (UINT8) images, pixel intensities span $[0, 255]$, mapping directly to $B = 256$ discrete bins.

When parallelizing histogram generation across thousands of concurrent GPU threads on SIMT (Single Instruction, Multiple Threads) architectures, multiple threads inevitably attempt to increment the same memory counter simultaneously. Under a straightforward parallelization scheme using **Global Memory Atomics**, concurrent write operations to the same global memory address are serialized by the hardware memory controller. Under skewed distributions or narrow bin counts, atomic serialization introduces massive stall cycles, severely bounding kernel throughput.

This report investigates the architectural mechanics and empirical performance of two CUDA implementations against a single-core CPU reference baseline:
1. **CUDA Global Atomics (`global_histogram.cu`):** Threads directly issue `atomicAdd()` instructions to a shared global histogram buffer residing in DRAM.
2. **CUDA Shared-Memory Privatization (`shared_histogram.cu`):** Each thread block maintains a private sub-histogram allocated in on-chip SRAM (`__shared__`). Threads execute fast, low-latency atomic operations locally within the block, followed by a parallel block-merge reduction into the global histogram buffer.

---

## 2. Hardware and Software Environment

All benchmarks were executed on bare-metal physical hardware:

| Component | Specification |
|:---|:---|
| **CPU** | 13th Gen Intel Core i5-13400 (10 Cores, 16 Threads, up to 4.60 GHz) |
| **Host Memory (RAM)** | 31.77 GiB DDR5 (~32 GB) |
| **GPU** | NVIDIA GeForce RTX 5060 Ti |
| **GPU Architecture** | Blackwell / Ada-Lovelace Generation (`sm_120`) |
| **Compute Capability** | 12.0 |
| **Video Memory (VRAM)** | 16,311 MiB (~16 GB GDDR6) |
| **Operating System** | Windows 11 Professional (Build 26200) |
| **CUDA Toolkit** | Version 13.5 (Driver Version 610.88, CUDA UMD 13.3) |
| **Host Compiler** | Microsoft Visual C++ (MSVC) Build Tools 2022 |
| **CUDA Compiler** | NVIDIA NVCC (`-std=c++17 -O2 -arch=sm_120`) |
| **Python Environment** | Python 3.13.17 / NumPy 2.5.5 / Pillow 13.0.0 / Matplotlib 3.10 |

---

## 3. Algorithmic Implementations

### 3.1 CPU Reference Baseline (`cpu_histogram.cpp`)
The CPU implementation processes input array $X$ sequentially:
$$\text{hist}[X[i]] \leftarrow \text{hist}[X[i]] + 1 \quad \forall i \in [0, N-1]$$
It executes with zero synchronization overhead but is constrained by single-core scalar throughput and host DRAM memory latency. It serves as the deterministic ground truth for correctness verification.

### 3.2 CUDA Global Atomics (`global_histogram.cu`)
Each GPU thread computes its global thread index:
$$\text{tid} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$
Threads process the data with grid-stride loops and issue atomic additions directly to global memory:
```cuda
__global__ void histogram_global_atomic(const uint8_t* __restrict__ data, int* __restrict__ hist, size_t N) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = tid; i < N; i += stride) {
        atomicAdd(&hist[data[i]], 1);
    }
}
```
*Limitation:* When multiple warps across different streaming multiprocessors (SMs) address the same bin (e.g., bin 128), hardware memory controllers serialize the requests, causing high memory replay cycles.

### 3.3 CUDA Shared-Memory Privatization (`shared_histogram.cu`)
Privatization eliminates cross-block global memory interference by isolating updates inside on-chip shared memory:
1. **Shared Histogram Allocation & Zero-Initialization:** Threads cooperatively clear $B = 256$ shared bins.
2. **Synchronize:** `__syncthreads()` ensures initialization is complete.
3. **Privatized Local Accumulation:** Threads update the local `s_hist` array via atomic operations in low-latency shared SRAM.
4. **Synchronize:** `__syncthreads()` ensures all local additions are finalized.
5. **Global Block-Merge Reduction:** Threads cooperatively write partial block histograms to global memory using a single atomic addition per bin:
```cuda
__global__ void histogram_shared_privatized(const uint8_t* __restrict__ data, int* __restrict__ hist, size_t N) {
    __shared__ int s_hist[256];
    int tid = threadIdx.x;
    if (tid < 256) s_hist[tid] = 0;
    __syncthreads();

    size_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = global_tid; i < N; i += stride) {
        atomicAdd(&s_hist[data[i]], 1);
    }
    __syncthreads();

    if (tid < 256) {
        atomicAdd(&hist[tid], s_hist[tid]);
    }
}
```

---

## 4. Empirical Evaluation & Experimental Results

### 4.1 Experiment 1: Baseline 1M Float Validation
- **Workload:** $N = 1,048,576$ elements, $B = 256$ bins, 4096 blocks, 256 threads/block.
- **Results:**
  - CPU Reference: $0.8361\text{ ms}$
  - Global Atomic GPU: $0.1562\text{ ms}$ ($5.35\times$ speedup)
  - Shared Privatized GPU: $0.0126\text{ ms}$ ($66.19\times$ speedup)
  - **Shared vs Global Improvement:** $12.36\times$
  - Correctness: 100% exact match ($1,048,576$ total elements).

### 4.2 Experiment 2: Synthetic UINT8 Contention Scaling ($N = 2^{24} = 16,777,216$)
To rigorously test atomic contention under controlled conditions, three distributions were evaluated on the RTX 5060 Ti:

| Distribution Pattern | Characteristics | CPU Time | Global Atomic | Shared Privatized | Global Speedup | Shared Speedup | Shared/Global Improvement |
|:---|:---|---:|---:|---:|---:|---:|---:|
| **Uniform** | Equal probability across 256 bins ($H = 8.000\text{ bits}$) | $4.4961\text{ ms}$ | $2.4348\text{ ms}$ | $0.0518\text{ ms}$ | $1.85\times$ | $86.71\times$ | **46.96×** |
| **Gaussian** | $\mu=128, \sigma=30$, clamped to $[0, 255]$ ($H = 6.954\text{ bits}$) | $5.0615\text{ ms}$ | $1.8216\text{ ms}$ | $0.0506\text{ ms}$ | $2.78\times$ | $99.99\times$ | **35.99×** |
| **Highly Skewed** | 90% mode at intensity 128, 10% uniform ($H = 1.264\text{ bits}$) | $23.6067\text{ ms}$ | $6.4014\text{ ms}$ | $0.0505\text{ ms}$ | $3.69\times$ | $467.79\times$ | **126.85×** |

*Key Insight:* Under 90% skew, global atomic execution time spikes from $2.43\text{ ms}$ to $6.40\text{ ms}$ due to severe serialization on bin 128. In contrast, shared privatization remains completely unaffected ($0.0505\text{ ms}$), providing a **126.85× performance boost** over global atomics.

### 4.3 Experiment 3: Bin-Count Scaling Analysis ($B \in \{16, 64, 256\}$)
Three full repeated benchmark runs (10 GPU iterations each) quantified the trade-off between privatization overhead and global atomic contention:

| Distribution | Bins ($B$) | Global Mean (ms) | Shared Mean (ms) | Speedup Ratio | Merge Updates per Grid ($4096 \times B$) |
|:---|---:|---:|---:|---:|---:|
| **Uniform** | 16 | 1.8588 | 0.0291 | 63.80× | 65,536 |
| **Uniform** | 64 | 2.7673 | 0.0311 | 88.98× | 262,144 |
| **Uniform** | 256 | 2.4463 | 0.0481 | 50.82× | 1,048,576 |
| **Gaussian** | 16 | 3.1218 | 0.0276 | 113.25× | 65,536 |
| **Gaussian** | 64 | 2.3485 | 0.0302 | 77.85× | 262,144 |
| **Gaussian** | 256 | 1.8364 | 0.0466 | 39.41× | 1,048,576 |
| **Highly Skewed** | 16 | 6.2244 | 0.0275 | 226.62× | 65,536 |
| **Highly Skewed** | 64 | 6.5892 | 0.0311 | 211.87× | 262,144 |
| **Highly Skewed** | 256 | 6.8695 | 0.0460 | 149.34× | 1,048,576 |

*Analysis of Overhead:* As bin count scales from 16 to 256, shared-memory kernel duration increases from $0.027\text{ ms}$ to $0.048\text{ ms}$. This reflects the cost of zeroing larger shared-memory allocations and performing $16\times$ more global atomic updates during the final block-merge phase ($1,048,576$ updates vs $65,536$).

---

## 5. Large-Scale Real-World Workload: DIV2K Dataset

The DIV2K high-resolution image dataset (800 training images, 2.23 billion pixels, **6,691,265,280 raw RGB bytes**) was benchmarked in 64 MiB streaming chunks.

### 5.1 DIV2K Multi-Run Statistical Results

| Execution Metric | Run 1 (ms) | Run 2 (ms) | Run 3 (ms) | Mean (ms) | Std Dev (ms) | CV (%) | Speedup vs CPU |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **CPU Time** | 5868.113 | 5820.028 | 5872.064 | 5853.402 | 28.970 | 0.495% | 1.00× |
| **Global Atomic Kernel** | 736.054 | 734.250 | 740.560 | 736.955 | 3.250 | 0.441% | 7.94× |
| **Shared Privatized Kernel** | 50.255 | 50.085 | 51.569 | 50.636 | 0.812 | 1.604% | **115.60×** |
| **Global Wall Time** | 5522.351 | 5434.680 | 5606.187 | 5521.073 | 85.761 | 1.553% | 1.06× |
| **Shared Wall Time** | 4848.474 | 4768.383 | 4779.363 | 4798.740 | 43.419 | 0.905% | 1.22× |

- **Pure Kernel Acceleration:** Shared privatization achieves **115.60× speedup over CPU** and **14.55× speedup over GPU global atomics**.
- **End-to-End Wall Time:** Host-device memory transfer (PCIe Gen4) and disk streaming dominate end-to-end latency ($~4.8\text{ s}$ total). Host-to-Device transfer took $977.34\text{ ms}$, Device-to-Host transfer took $3.01\text{ ms}$, confirming that kernel optimization transforms the workload from compute/contention-bound to I/O-bound.

### 5.2 DIV2K Intensity Distribution Statistics
Processed via 48 MiB memory-mapped chunks (`analyze_div2k.py`):
- **Total Samples:** 6,691,265,280 (Red: 2.23B, Green: 2.23B, Blue: 2.23B)
- **Combined RGB Entropy:** $7.9390\text{ bits}$ (close to maximum theoretical 8-bit entropy of $8.0\text{ bits}$)
- **Mean Intensity:** $109.6504$ | **Std Dev:** $72.1351$ | **CV:** $0.3040$
- **Most Frequent Intensity:** Bin 0 ($93,426,174$ occurrences, $1.40\%$ of dataset)
- **Channel Means:** Red: $114.45$, Green: $111.47$, Blue: $103.03$

---

## 6. AI Application Extensions

### 6.1 INT8 Quantization Calibration (`quantization_calibration.cu`)
Histograms enable post-training quantization calibration for tensor activations:
1. Construct activation absolute magnitude histogram.
2. Determine percentile threshold $\alpha \in \{99\%, 99.9\%, 99.99\%\}$.
3. Scale factor: $S = \frac{\alpha}{127}$.
4. Symmetric quantize: $q = \text{clamp}(\text{round}(x / S), -128, 127)$.
5. Dequantize: $\hat{x} = q \times S$.

*Empirical Results:* On Gaussian activations, clipping at 99.99% percentile reduced Mean Squared Error (MSE) from $0.001815$ (at 99%) down to $0.000079$, reducing saturation from $0.998\%$ to $0.0098\%$.

### 6.2 GPU Parallel CDF and Percentile Extraction (`gpu_parallel_cdf.cu`)
Implements an on-device Cumulative Distribution Function via a 256-thread Hillis-Steele inclusive scan across 8 iterations in shared memory:
- GPU Histogram Kernel: $0.018785\text{ ms}$
- Parallel CDF Scan Kernel: $0.006907\text{ ms}$
- Percentile Extraction Kernel: $0.013688\text{ ms}$
- Verification: 100% exact numerical match with CPU prefix-sum CDF across all tested distributions.

---

## 7. Conclusions

1. **Atomic Contention is a Critical Bottleneck:** Global atomics degrade significantly under non-uniform distributions (up to $3.6\times$ latency increase under 90% skew).
2. **Privatization Delivers Linear Scaling:** Shared memory isolates atomic updates within L1/shared SRAM, providing up to **126.85× speedup** over global atomics under heavy skew, and **115.60× speedup over CPU** on the 6.69-billion-value DIV2K image benchmark.
3. **Trade-offs of Bin Count:** Privatization is most effective when $B$ fits comfortably in shared memory ($B \le 256$); larger bin counts incur proportional overhead during zeroing and block-merge reductions.
