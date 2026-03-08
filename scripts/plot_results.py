#!/usr/bin/env python3
"""
plot_results.py — Performance Visualisation for N-Body Benchmarks

Reads benchmark.csv and generates publication-quality plots:
  1. Execution Time vs Problem Size (N)
  2. Speedup vs Number of Threads
  3. Efficiency vs Number of Threads
  4. Execution Time vs Threads (for fixed N)

Usage:
    python3 scripts/plot_results.py [--input results/benchmark.csv] [--outdir results/]
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

# ---- Matplotlib Configuration ------------------------------------------------

matplotlib.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
})

# Colour palette for paradigms
COLOURS = {
    'serial':   '#2c3e50',
    'openmp':   '#e74c3c',
    'pthreads': '#3498db',
    'mpi':      '#2ecc71',
    'cuda':     '#f39c12',
    'hybrid':   '#9b59b6',
}

MARKERS = {
    'serial':   'o',
    'openmp':   's',
    'pthreads': '^',
    'mpi':      'D',
    'cuda':     '*',
    'hybrid':   'P',
}


def load_data(filepath):
    """Load benchmark CSV into a DataFrame."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df


def plot_time_vs_n(df, outdir):
    """Plot execution time vs problem size for all paradigms."""
    fig, ax = plt.subplots()

    for paradigm in df['paradigm'].unique():
        subset = df[df['paradigm'] == paradigm]
        # For multi-threaded paradigms, pick the best (max threads) result
        if paradigm != 'serial':
            subset = subset.loc[subset.groupby('N')['time_sec'].idxmin()]
        ax.plot(subset['N'], subset['time_sec'],
                marker=MARKERS.get(paradigm, 'o'),
                color=COLOURS.get(paradigm, '#333'),
                label=paradigm.capitalize(),
                linewidth=2, markersize=8)

    ax.set_xlabel('Number of Bodies (N)')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Execution Time vs Problem Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    path = os.path.join(outdir, 'time_vs_n.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_speedup_vs_threads(df, outdir):
    """Plot speedup vs threads for each problem size."""
    serial_times = df[df['paradigm'] == 'serial'].set_index('N')['time_sec']
    parallel = df[df['paradigm'] != 'serial']

    problem_sizes = sorted(parallel['N'].unique())

    fig, axes = plt.subplots(1, len(problem_sizes),
                              figsize=(6 * len(problem_sizes), 5),
                              squeeze=False)

    for idx, n_val in enumerate(problem_sizes):
        ax = axes[0][idx]
        serial_t = serial_times.get(n_val, None)
        if serial_t is None:
            continue

        sub = parallel[parallel['N'] == n_val]

        for paradigm in sub['paradigm'].unique():
            ps = sub[sub['paradigm'] == paradigm].sort_values('threads')
            speedup = serial_t / ps['time_sec']
            ax.plot(ps['threads'], speedup,
                    marker=MARKERS.get(paradigm, 'o'),
                    color=COLOURS.get(paradigm, '#333'),
                    label=paradigm.capitalize(),
                    linewidth=2, markersize=8)

        # Ideal speedup line
        max_t = sub['threads'].max()
        ideal = range(1, int(max_t) + 1)
        ax.plot(ideal, ideal, '--', color='gray', alpha=0.5, label='Ideal')

        ax.set_xlabel('Threads / Processes')
        ax.set_ylabel('Speedup')
        ax.set_title(f'N = {n_val}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Speedup vs Threads', fontsize=16, y=1.02)
    path = os.path.join(outdir, 'speedup_vs_threads.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_efficiency_vs_threads(df, outdir):
    """Plot parallel efficiency vs threads."""
    serial_times = df[df['paradigm'] == 'serial'].set_index('N')['time_sec']
    parallel = df[df['paradigm'] != 'serial']

    fig, ax = plt.subplots()

    # Use the largest problem size for the efficiency plot
    max_n = parallel['N'].max()
    serial_t = serial_times.get(max_n, None)
    if serial_t is None:
        print("  Warning: no serial time for largest N, skipping efficiency plot.")
        return

    sub = parallel[parallel['N'] == max_n]

    for paradigm in sub['paradigm'].unique():
        ps = sub[sub['paradigm'] == paradigm].sort_values('threads')
        speedup = serial_t / ps['time_sec']
        efficiency = speedup / ps['threads']
        ax.plot(ps['threads'], efficiency,
                marker=MARKERS.get(paradigm, 'o'),
                color=COLOURS.get(paradigm, '#333'),
                label=paradigm.capitalize(),
                linewidth=2, markersize=8)

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Ideal')
    ax.set_xlabel('Threads / Processes')
    ax.set_ylabel('Efficiency (Speedup / Threads)')
    ax.set_title(f'Parallel Efficiency (N = {max_n})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.5)

    path = os.path.join(outdir, 'efficiency_vs_threads.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_time_vs_threads(df, outdir):
    """Plot execution time vs threads for all paradigms at fixed N."""
    parallel = df[df['paradigm'] != 'serial']
    max_n = parallel['N'].max()
    sub = parallel[parallel['N'] == max_n]

    fig, ax = plt.subplots()

    # Add serial baseline as horizontal line
    serial_t = df[(df['paradigm'] == 'serial') & (df['N'] == max_n)]
    if not serial_t.empty:
        ax.axhline(y=serial_t['time_sec'].values[0], color=COLOURS['serial'],
                    linestyle='--', linewidth=2, label='Serial', alpha=0.7)

    for paradigm in sub['paradigm'].unique():
        ps = sub[sub['paradigm'] == paradigm].sort_values('threads')
        ax.plot(ps['threads'], ps['time_sec'],
                marker=MARKERS.get(paradigm, 'o'),
                color=COLOURS.get(paradigm, '#333'),
                label=paradigm.capitalize(),
                linewidth=2, markersize=8)

    ax.set_xlabel('Threads / Processes')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title(f'Execution Time vs Threads (N = {max_n})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(outdir, 'time_vs_threads.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='Plot N-Body benchmark results')
    parser.add_argument('--input', default='results/benchmark.csv',
                        help='Path to benchmark CSV')
    parser.add_argument('--outdir', default='results/',
                        help='Directory to save plots')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: '{args.input}' not found. Run benchmarks first.")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    print("Loading benchmark data...")
    df = load_data(args.input)
    print(f"  {len(df)} records from {df['paradigm'].nunique()} paradigms\n")

    print("Generating plots:")
    plot_time_vs_n(df, args.outdir)
    plot_speedup_vs_threads(df, args.outdir)
    plot_efficiency_vs_threads(df, args.outdir)
    plot_time_vs_threads(df, args.outdir)

    print(f"\nAll plots saved to {args.outdir}")


if __name__ == '__main__':
    main()
