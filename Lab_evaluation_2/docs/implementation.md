# Implementation guide

- `cpu_histogram.cpp`: CPU uint8 reference example.
- `global_histogram.cu`: float histogram using global atomics.
- `shared_histogram.cu`: float histogram using block-local shared histograms.
- `uint8_distribution_experiment.cu`: core 2^24 uint8 benchmark across uniform, Gaussian, and skewed data.
- `uint8_bin_count_experiment.cu`: 16/64/256-bin comparison.
- `distribution_experiment.cu`: four float distributions.
- `quantization_calibration.cu`: histogram-based percentile calibration prototype.
- `gpu_parallel_cdf.cu`: GPU histogram, inclusive scan, percentile extraction.
- `div2k_histogram_benchmark.cu`: chunked RGB byte histogram for a preprocessed DIV2K binary.
- Python scripts prepare and analyze DIV2K and summarize result CSVs.
