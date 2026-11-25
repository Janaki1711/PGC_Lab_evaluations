import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set professional aesthetics
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

results_dir = Path("results")
graphs_dir = Path("graphs")
results_dir.mkdir(exist_ok=True)
graphs_dir.mkdir(exist_ok=True)

def save_fig(fig, filename):
    for d in [results_dir, graphs_dir]:
        fig.savefig(d / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}")

# ==============================================================================
# Graph 1: div2k_kernel_comparison.png
# Execution time comparison between Global Atomics vs Shared Memory on DIV2K (6.69B elements)
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 5))
methods = ['Global Atomic GPU', 'Shared-Memory GPU']
means = [736.955, 50.636]
std_devs = [3.250, 0.812]
colors = ['#1f77b4', '#2ca02c']

bars = ax.bar(methods, means, yerr=std_devs, capsize=8, color=colors, width=0.45, edgecolor='#222222', alpha=0.9)
ax.set_ylabel('Kernel Execution Time (ms)', fontsize=12, fontweight='bold')
ax.set_title('DIV2K Benchmark: Kernel Execution Time\n(6.69 Billion Values, 256 Bins, RTX 5060 Ti)', fontsize=13, fontweight='bold', pad=15)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_ylim(0, 850)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 18, f"{yval:.2f} ms", ha='center', va='bottom', fontsize=11, fontweight='bold')

# Speedup annotation
ax.annotate('14.55× Kernel Speedup\n(Privatization Benefit)', 
            xy=(1, 50.636), xytext=(0.55, 380),
            arrowprops=dict(facecolor='#d62728', shrink=0.08, width=2, headwidth=8),
            fontsize=11, fontweight='bold', color='#b22222',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff2f2', edgecolor='#d62728', lw=1.2))

save_fig(fig, 'div2k_kernel_comparison.png')

# ==============================================================================
# Graph 2: div2k_cpu_gpu_comparison.png
# Kernel Speedup vs CPU Reference on DIV2K
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 5))
labels = ['CPU Baseline\n(i5-13400)', 'Global Atomic GPU\n(RTX 5060 Ti)', 'Shared Privatized GPU\n(RTX 5060 Ti)']
speedups = [1.0, 7.94, 115.60]
colors = ['#7f7f7f', '#1f77b4', '#2ca02c']

bars = ax.bar(labels, speedups, color=colors, width=0.5, edgecolor='#222222', alpha=0.9)
ax.set_ylabel('Speedup Factor (relative to CPU)', fontsize=12, fontweight='bold')
ax.set_title('DIV2K Histogram: Acceleration Relative to Single-Core CPU Baseline', fontsize=13, fontweight='bold', pad=15)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_yscale('log')
ax.set_ylim(0.5, 250)

# Format y-axis ticks for log scale
ax.set_yticks([1, 5, 10, 50, 100, 200])
ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval * 1.15, f"{yval:.2f}×", ha='center', va='bottom', fontsize=11, fontweight='bold')

save_fig(fig, 'div2k_cpu_gpu_comparison.png')

# ==============================================================================
# Graph 3: div2k_wall_comparison.png
# End-to-End Wall Time across 3 Repeated Executions
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 5.5))
runs = ['Run 1', 'Run 2', 'Run 3']
x = np.arange(len(runs))
width = 0.25

cpu_wall = [5868.113, 5820.028, 5872.064]
global_wall = [5522.351, 5434.680, 5606.187]
shared_wall = [4848.474, 4768.383, 4779.363]

r1 = ax.bar(x - width, cpu_wall, width, label='CPU Reference', color='#7f7f7f', edgecolor='#222222', alpha=0.9)
r2 = ax.bar(x, global_wall, width, label='Global Atomic GPU', color='#1f77b4', edgecolor='#222222', alpha=0.9)
r3 = ax.bar(x + width, shared_wall, width, label='Shared-Memory GPU', color='#2ca02c', edgecolor='#222222', alpha=0.9)

ax.set_ylabel('End-to-End Wall Time (ms)', fontsize=12, fontweight='bold')
ax.set_title('DIV2K Benchmark: End-to-End Wall Time across Repeated Executions\n(Includes Host-Device I/O, 64 MiB Chunk Transfers & Aggregation)', fontsize=12, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(runs, fontsize=11, fontweight='bold')
ax.legend(frameon=True, facecolor='white', framealpha=0.9)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_ylim(0, 6800)

for rects in [r1, r2, r3]:
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.0, h + 80, f"{h:.0f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

save_fig(fig, 'div2k_wall_comparison.png')

# ==============================================================================
# Graph 4: bin_count_uniform.png
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 5))
bins = [16, 64, 256]
global_uniform = [1.8588, 2.7673, 2.4463]
shared_uniform = [0.0291, 0.0311, 0.0481]

ax.plot(bins, global_uniform, marker='o', linewidth=2.5, markersize=8, color='#1f77b4', label='Global Atomics (mean)')
ax.plot(bins, shared_uniform, marker='s', linewidth=2.5, markersize=8, color='#2ca02c', label='Shared Privatization (mean)')

for b, g, s in zip(bins, global_uniform, shared_uniform):
    ax.annotate(f"{g:.4f} ms", (b, g), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold', color='#1f77b4')
    ax.annotate(f"{s:.4f} ms", (b, s), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold', color='#2ca02c')

ax.set_xscale('log', base=2)
ax.set_xticks([16, 64, 256])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xlabel('Histogram Bins', fontsize=12, fontweight='bold')
ax.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
ax.set_title('Uniform Distribution: Execution Time vs Bin Count\n(2^24 Elements, 4096 Blocks, 3-Run Average)', fontsize=13, fontweight='bold', pad=15)
ax.legend(frameon=True, loc='center right')
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_ylim(0, 3.4)

save_fig(fig, 'bin_count_uniform.png')

# ==============================================================================
# Graph 5: bin_count_gaussian.png
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 5))
global_gaussian = [3.1218, 2.3485, 1.8364]
shared_gaussian = [0.0276, 0.0302, 0.0466]

ax.plot(bins, global_gaussian, marker='o', linewidth=2.5, markersize=8, color='#1f77b4', label='Global Atomics (mean)')
ax.plot(bins, shared_gaussian, marker='s', linewidth=2.5, markersize=8, color='#2ca02c', label='Shared Privatization (mean)')

for b, g, s in zip(bins, global_gaussian, shared_gaussian):
    ax.annotate(f"{g:.4f} ms", (b, g), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold', color='#1f77b4')
    ax.annotate(f"{s:.4f} ms", (b, s), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold', color='#2ca02c')

ax.set_xscale('log', base=2)
ax.set_xticks([16, 64, 256])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xlabel('Histogram Bins', fontsize=12, fontweight='bold')
ax.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
ax.set_title('Gaussian Distribution: Execution Time vs Bin Count\n(2^24 Elements, μ=128, σ=30, 3-Run Average)', fontsize=13, fontweight='bold', pad=15)
ax.legend(frameon=True, loc='center right')
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_ylim(0, 3.6)

save_fig(fig, 'bin_count_gaussian.png')

# ==============================================================================
# Graph 6: bin_count_highly_skewed.png
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 5))
global_skewed = [6.2244, 6.5892, 6.8695]
shared_skewed = [0.0275, 0.0311, 0.0460]

ax.plot(bins, global_skewed, marker='o', linewidth=2.5, markersize=8, color='#d62728', label='Global Atomics (High Contention)')
ax.plot(bins, shared_skewed, marker='s', linewidth=2.5, markersize=8, color='#2ca02c', label='Shared Privatization (Contention-Immune)')

for b, g, s in zip(bins, global_skewed, shared_skewed):
    ax.annotate(f"{g:.4f} ms", (b, g), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold', color='#d62728')
    ax.annotate(f"{s:.4f} ms", (b, s), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold', color='#2ca02c')

ax.set_xscale('log', base=2)
ax.set_xticks([16, 64, 256])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xlabel('Histogram Bins', fontsize=12, fontweight='bold')
ax.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
ax.set_title('Highly Skewed Distribution: Execution Time vs Bin Count\n(2^24 Elements, 90% Mode at 128, 3-Run Average)', fontsize=13, fontweight='bold', pad=15)
ax.legend(frameon=True, loc='center right')
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_ylim(0, 7.8)

save_fig(fig, 'bin_count_highly_skewed.png')

# ==============================================================================
# Graph 7: bin_count_speedup_comparison.png
# Global/Shared Ratio across Distributions and Bin Counts
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(len(bins))
width = 0.25

uniform_ratios = [63.80, 88.98, 50.82]
gaussian_ratios = [113.25, 77.85, 39.41]
skewed_ratios = [226.62, 211.87, 149.34]

b1 = ax.bar(x - width, uniform_ratios, width, label='Uniform', color='#1f77b4', edgecolor='#222222', alpha=0.9)
b2 = ax.bar(x, gaussian_ratios, width, label='Gaussian', color='#ff7f0e', edgecolor='#222222', alpha=0.9)
b3 = ax.bar(x + width, skewed_ratios, width, label='Highly Skewed (90% Mode)', color='#d62728', edgecolor='#222222', alpha=0.9)

ax.set_ylabel('Speedup Ratio (Global ms / Shared ms)', fontsize=12, fontweight='bold')
ax.set_xlabel('Histogram Bins', fontsize=12, fontweight='bold')
ax.set_title('Privatization Improvement Ratio (Global / Shared Time)\nAcross Input Distributions and Bin Counts', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels([f"{b} Bins" for b in bins], fontsize=11, fontweight='bold')
ax.legend(frameon=True, facecolor='white', framealpha=0.9)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_ylim(0, 260)

for rects in [b1, b2, b3]:
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.0, h + 3, f"{h:.1f}×", ha='center', va='bottom', fontsize=9, fontweight='bold')

save_fig(fig, 'bin_count_speedup_comparison.png')
print("All 7 performance graphs successfully generated!")
