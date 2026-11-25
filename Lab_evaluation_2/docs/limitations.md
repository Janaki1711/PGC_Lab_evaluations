# Limitations and provenance

- These are reconstructed implementations, not the original source files.
- Historical measurements were recorded on an RTX 5060 Ti system with CUDA 13.5 and are retained as historical context only.
- Reconstructed programs have not been executed as part of this package creation.
- GPU timings depend on clocks, drivers, compiler, temperature, workload, and timing boundaries.
- Kernel-only time is not end-to-end application time.
- The DIV2K dataset is not bundled; download it separately.
- Float distribution, CDF, and quantization programs are extensions to the core uint8 histogram assignment.
- Some extension programs are simplified research prototypes and should be validated and extended before being used to support quantitative claims.

The DIV2K CUDA sample prioritizes chunked correctness and bounded device memory. Its current timing is end-to-end wall time; it does not yet provide the historical separate per-kernel and transfer timing breakdown. Rework timing instrumentation before making direct timing comparisons.
