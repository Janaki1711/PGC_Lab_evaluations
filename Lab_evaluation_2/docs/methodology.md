# Methodology

A histogram maps each input value to a bin and increments that bin's count. In a parallel GPU implementation, multiple threads can update the same bin concurrently, requiring atomic operations.

## Global atomics
Every input thread performs an atomic increment directly against the global histogram. This is simple but can serialize heavily when many inputs map to the same bin.

## Shared-memory privatization
Each CUDA block creates a private histogram in shared memory. Threads atomically update the local histogram. After synchronization, the block merges its local bin counts into the global histogram. This reduces global atomic traffic, although merge cost and shared-memory initialization grow with the number of bins.

## Correctness
The sum of all histogram bins must equal the number of processed values. GPU output should also match an independently computed CPU reference.
