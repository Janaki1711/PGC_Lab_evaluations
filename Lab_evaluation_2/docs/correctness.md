# Correctness validation

1. Compute a CPU reference histogram.
2. Compute the corresponding GPU histogram.
3. Compare every bin.
4. Verify that the sum of bins equals the number of input values.
5. For chunked datasets, accumulate partial counts in 64-bit host counters to prevent overflow.
6. Validate boundary mapping for minimum and maximum input values.
