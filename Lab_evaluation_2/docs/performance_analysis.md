# Performance analysis

Global atomic performance is sensitive to the distribution of values because concentrated inputs increase contention on a small number of bins. Shared-memory privatization moves most updates to block-local storage and merges partial histograms afterward.

Increasing bin count can increase shared histogram initialization and merge work. Results are implementation- and workload-dependent; do not infer universal monotonic trends.

Report kernel timing separately from host-to-device transfer, device-to-host transfer, file I/O, and total wall-clock time.
