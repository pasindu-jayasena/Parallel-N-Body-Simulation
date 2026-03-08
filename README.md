# Parallel N-Body Simulation

A high-performance gravitational N-Body simulation implemented in **six parallel programming paradigms** for comparative performance analysis.

| Paradigm | File | Description |
|----------|------|-------------|
| **Serial** | `src/serial.c` | O(N²) baseline with Newton's third law optimisation |
| **OpenMP** | `src/openmp.c` | Shared-memory with `#pragma omp parallel for` |
| **Pthreads** | `src/pthreads.c` | Manual threading with barrier synchronisation |
| **MPI** | `src/mpi.c` | Distributed-memory with `MPI_Allgather` |
| **CUDA** | `src/cuda_nbody.cu` | GPU acceleration with shared-memory tiling |
| **Hybrid** | `src/hybrid.c` | MPI + OpenMP combined model |

## Project Structure

```
include/           Header files (types, I/O, common utilities)
src/               Source files (one per paradigm + shared common.c)
scripts/           Benchmark runner, plotting, RMSE verification
results/           Output CSV files and performance plots
build/             Compiled binaries (created by make)
```

## Prerequisites

- **GCC** with OpenMP support (`-fopenmp`)
- **POSIX Threads** (standard on Linux)
- **MPI** (OpenMPI or MPICH): `sudo apt install libopenmpi-dev`
- **CUDA Toolkit** (optional): `nvcc` compiler
- **Python 3** with `matplotlib`, `pandas`, `numpy` (for plotting)

## Build

### Using Make (recommended)

```bash
# Build all targets (excluding CUDA)
make all

# Build individual targets
make serial
make openmp
make pthreads
make mpi
make hybrid
make cuda        # requires nvcc

# Clean
make clean
```

### Using CMake

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

## Run

```bash
# Usage:  ./build/nbody_<paradigm> <N> <steps> [threads]

# Serial baseline
./build/nbody_serial 1000 100

# OpenMP (4 threads)
OMP_NUM_THREADS=4 ./build/nbody_openmp 1000 100

# Pthreads (4 threads)
./build/nbody_pthreads 1000 100 4

# MPI (4 processes)
mpirun -np 4 ./build/nbody_mpi 1000 100

# CUDA
./build/nbody_cuda 1000 100

# Hybrid (2 MPI procs × 2 OMP threads each)
OMP_NUM_THREADS=2 mpirun -np 2 ./build/nbody_hybrid 1000 100
```

## Benchmark

```bash
# Run automated benchmarks (saves to results/benchmark.csv)
bash scripts/run_benchmarks.sh

# Generate performance plots
python3 scripts/plot_results.py

# Verify accuracy (RMSE against serial baseline)
python3 scripts/verify_accuracy.py
```

## Output Format

Each program prints timing in the format:
```
[Paradigm  ] N=1000   Steps=100  Time=1.2345 s
```

Final body positions are saved to `results/<paradigm>_output.csv`.

## Evaluation Metrics

- **Execution Time** (seconds)
- **Speedup** = Serial Time / Parallel Time
- **Efficiency** = Speedup / Number of Threads
- **RMSE** for numerical accuracy against serial baseline
