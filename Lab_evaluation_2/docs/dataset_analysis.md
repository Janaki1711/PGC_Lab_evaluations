# DIV2K dataset analysis

The pipeline converts each DIV2K high-resolution training image to RGB and concatenates raw interleaved uint8 values. The manifest records each image's dimensions and byte count.

The analysis computes combined RGB and per-channel histograms, mean, standard deviation, entropy, coefficient of variation, and bin population statistics. Chunk processing and memory mapping avoid loading the full multi-gigabyte dataset into RAM at once.
