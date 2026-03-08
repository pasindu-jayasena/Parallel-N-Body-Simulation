#!/usr/bin/env python3
"""
verify_accuracy.py — RMSE Verification for N-Body Simulation

Compares the final positions from each parallel implementation against
the serial baseline output. Computes Root Mean Square Error (RMSE) and
reports pass/fail status.

Usage:
    python3 scripts/verify_accuracy.py [--baseline results/serial_output.csv]
                                       [--results-dir results/]

Expected CSV format (header + data):
    x,y,z,vx,vy,vz,mass
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


# Maximum acceptable RMSE (should be ~0 for identical algorithms)
RMSE_THRESHOLD = 1e-6


def compute_rmse(baseline, test):
    """Compute RMSE of positions between baseline and test DataFrames."""
    dx = baseline['x'].values - test['x'].values
    dy = baseline['y'].values - test['y'].values
    dz = baseline['z'].values - test['z'].values

    mse = np.mean(dx**2 + dy**2 + dz**2)
    return np.sqrt(mse)


def main():
    parser = argparse.ArgumentParser(description='Verify N-Body accuracy via RMSE')
    parser.add_argument('--baseline', default='results/serial_output.csv',
                        help='Path to serial baseline CSV')
    parser.add_argument('--results-dir', default='results/',
                        help='Directory containing parallel output CSVs')
    args = parser.parse_args()

    # Load baseline
    if not os.path.exists(args.baseline):
        print(f"Error: baseline '{args.baseline}' not found.")
        print("Run the serial simulation first: ./build/nbody_serial <N> <steps>")
        sys.exit(1)

    baseline = pd.read_csv(args.baseline)
    n = len(baseline)
    print(f"Baseline: {args.baseline} ({n} bodies)\n")

    # Paradigms and their expected output files
    paradigms = {
        'OpenMP':   'openmp_output.csv',
        'Pthreads': 'pthreads_output.csv',
        'MPI':      'mpi_output.csv',
        'CUDA':     'cuda_output.csv',
        'Hybrid':   'hybrid_output.csv',
    }

    print(f"{'Paradigm':<12} {'RMSE':<20} {'Status':<10}")
    print("-" * 42)

    all_pass = True

    for name, filename in paradigms.items():
        filepath = os.path.join(args.results_dir, filename)

        if not os.path.exists(filepath):
            print(f"{name:<12} {'(file not found)':<20} {'SKIP':<10}")
            continue

        test = pd.read_csv(filepath)

        if len(test) != n:
            print(f"{name:<12} {'(size mismatch)':<20} {'FAIL':<10}")
            all_pass = False
            continue

        rmse = compute_rmse(baseline, test)
        status = 'PASS' if rmse < RMSE_THRESHOLD else 'FAIL'
        if status == 'FAIL':
            all_pass = False

        print(f"{name:<12} {rmse:<20.6e} {status:<10}")

    print("-" * 42)
    if all_pass:
        print("\nAll paradigms PASSED accuracy verification.")
    else:
        print("\nSome paradigms FAILED. Check the RMSE values above.")

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
