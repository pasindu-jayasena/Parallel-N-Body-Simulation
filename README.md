# Parallel N-Body Simulation
## EE7218/EC7207: High Performance Computing

---

## Project Structure

```
HPCPresentation/
├── README.md               ← This file (full documentation)
├── include/
│   └── nbody.h             ← Shared types and constants
└── src/
    ├── common.c            ← Shared utility functions (used by ALL programs)
    ├── serial.c            ← Deliverable 1: Serial Code
    ├── openmp.c            ← Deliverable 2: Shared Memory (OpenMP)
    ├── pthreads.c          ← Deliverable 2: Shared Memory (POSIX Threads)
    ├── mpi_nbody.c         ← Deliverable 3: Distributed Memory (MPI)
    ├── cuda_nbody.cu       ← CUDA GPU version
    └── hybrid.c            ← Deliverable 4: Hybrid (MPI + OpenMP)
```

---

## Prerequisites

Install these tools in WSL (Ubuntu):

```bash
sudo apt update
sudo apt install gcc make openmpi-bin libopenmpi-dev
```

Verify installation:
```bash
gcc --version       # Should show GCC version
mpicc --version     # Should show MPI compiler version
```

---

## Open Project in WSL

```bash
cd /mnt/c/Users/CHAMODYA/Desktop/HPCPresentation
```

---

## Step 1 — Compile

Run each line separately or all at once:

### Compile Serial
```bash
gcc -O2 -Wall -std=c99 -Iinclude -o nbody_serial src/serial.c src/common.c -lm
```

### Compile OpenMP
```bash
gcc -O2 -Wall -std=c99 -Iinclude -fopenmp -o nbody_openmp src/openmp.c src/common.c -lm
```

### Compile POSIX Threads
```bash
gcc -O2 -Wall -std=c99 -Iinclude -pthread -o nbody_pthreads src/pthreads.c src/common.c -lm
```

### Compile MPI
```bash
mpicc -O2 -Wall -std=c99 -Iinclude -o nbody_mpi src/mpi_nbody.c src/common.c -lm
```

### Compile Hybrid (MPI + OpenMP)
```bash
mpicc -O2 -Wall -std=c99 -Iinclude -fopenmp -o nbody_hybrid src/hybrid.c src/common.c -lm
```

### Compile CUDA (only if NVIDIA GPU available)
```bash
nvcc -O2 -Iinclude -o nbody_cuda src/cuda_nbody.cu src/common.c -lm
```

### Compile ALL at once (copy-paste this block)
```bash
gcc  -O2 -Wall -std=c99 -Iinclude           -o nbody_serial   src/serial.c    src/common.c -lm
gcc  -O2 -Wall -std=c99 -Iinclude -fopenmp  -o nbody_openmp   src/openmp.c    src/common.c -lm
gcc  -O2 -Wall -std=c99 -Iinclude -pthread  -o nbody_pthreads src/pthreads.c  src/common.c -lm
mpicc -O2 -Wall -std=c99 -Iinclude          -o nbody_mpi      src/mpi_nbody.c src/common.c -lm
mpicc -O2 -Wall -std=c99 -Iinclude -fopenmp -o nbody_hybrid   src/hybrid.c    src/common.c -lm
echo "All compiled successfully!"
```

---

## Step 2 — Run

### Command Format
```
./program_name  <N>  <steps>  [threads]
```
| Argument | Description | Example |
|----------|-------------|---------|
| `N` | Number of bodies | `1000` |
| `steps` | Number of simulation timesteps | `100` |
| `threads` | Number of threads (parallel versions only) | `4` |

---

### 1. Run Serial (Deliverable 1)
```bash
./nbody_serial 1000 100
```

**Expected output:**
```
=== Serial N-Body Simulation ===
Bodies: 1000 | Steps: 100

Serial Time: 0.5465 seconds
```

---

### 2. Run OpenMP (Deliverable 2a — Shared Memory)

Change the last number to change how many threads are used:

```bash
./nbody_openmp 1000 100 1
./nbody_openmp 1000 100 2
./nbody_openmp 1000 100 4
./nbody_openmp 1000 100 8
```

**Expected output:**
```
=== OpenMP N-Body Simulation ===
Bodies: 1000 | Steps: 100 | Threads: 4

Serial Time:   0.5385 seconds
Parallel Time: 0.1822 seconds
Speedup:       2.96x
RMSE:          0.000000e+00
```

---

### 3. Run POSIX Threads (Deliverable 2b — Shared Memory)

```bash
./nbody_pthreads 1000 100 1
./nbody_pthreads 1000 100 2
./nbody_pthreads 1000 100 4
./nbody_pthreads 1000 100 8
```

**Expected output:**
```
=== POSIX Threads N-Body Simulation ===
Bodies: 1000 | Steps: 100 | Threads: 4

Serial Time:   0.4917 seconds
Parallel Time: 0.1927 seconds
Speedup:       2.55x
RMSE:          0.000000e+00
```

---

### 4. Run MPI (Deliverable 3 — Distributed Memory)

Change `-np` number to change how many processes are used:

```bash
mpirun --oversubscribe -np 1 ./nbody_mpi 1000 100
mpirun --oversubscribe -np 2 ./nbody_mpi 1000 100
mpirun --oversubscribe -np 4 ./nbody_mpi 1000 100
mpirun --oversubscribe -np 8 ./nbody_mpi 1000 100
```

**Expected output:**
```
=== MPI N-Body Simulation ===
Bodies: 1000 | Steps: 100 | Processes: 4

Serial Time:   0.4680 seconds
Parallel Time: 0.1605 seconds
Speedup:       2.92x
RMSE:          0.000000e+00
```

---

### 5. Run CUDA (GPU — only if NVIDIA GPU available)

```bash
./nbody_cuda 1000 100
```

**Expected output:**
```
=== CUDA N-Body Simulation ===
Bodies: 1000 | Steps: 100

Serial Time: 0.5300 seconds
CUDA Time:   0.0210 seconds
Speedup:     25.24x
RMSE:        0.000000e+00
```

---

### 6. Run Hybrid — MPI + OpenMP (Deliverable 4)

`OMP_NUM_THREADS` = OpenMP threads per MPI process
`-np` = number of MPI processes

```bash
OMP_NUM_THREADS=2 mpirun --oversubscribe -np 2 ./nbody_hybrid 1000 100
OMP_NUM_THREADS=2 mpirun --oversubscribe -np 4 ./nbody_hybrid 1000 100
OMP_NUM_THREADS=4 mpirun --oversubscribe -np 2 ./nbody_hybrid 1000 100
OMP_NUM_THREADS=4 mpirun --oversubscribe -np 4 ./nbody_hybrid 1000 100
```

**Expected output:**
```
=== Hybrid (MPI + OpenMP) N-Body Simulation ===
Bodies: 1000 | Steps: 100 | MPI: 2 | OMP: 2

Serial Time:   0.4788 seconds
Parallel Time: 0.1724 seconds
Speedup:       2.78x
RMSE:          0.000000e+00
```

---

## Step 3 — Test Different Problem Sizes (for Analysis Report)

Run each program with different N values to collect timing data:

```bash
# Serial — baseline timings
./nbody_serial  500  100
./nbody_serial  1000 100
./nbody_serial  2000 100
./nbody_serial  5000 100

# OpenMP — vary both N and threads
./nbody_openmp  500  100 1
./nbody_openmp  500  100 2
./nbody_openmp  500  100 4
./nbody_openmp  500  100 8
./nbody_openmp  1000 100 1
./nbody_openmp  1000 100 2
./nbody_openmp  1000 100 4
./nbody_openmp  1000 100 8
./nbody_openmp  2000 100 4
./nbody_openmp  5000 100 4

# Pthreads — vary both N and threads
./nbody_pthreads 1000 100 1
./nbody_pthreads 1000 100 2
./nbody_pthreads 1000 100 4
./nbody_pthreads 1000 100 8

# MPI — vary both N and processes
mpirun --oversubscribe -np 1 ./nbody_mpi 1000 100
mpirun --oversubscribe -np 2 ./nbody_mpi 1000 100
mpirun --oversubscribe -np 4 ./nbody_mpi 1000 100
mpirun --oversubscribe -np 8 ./nbody_mpi 1000 100

# Hybrid
OMP_NUM_THREADS=2 mpirun --oversubscribe -np 2 ./nbody_hybrid 1000 100
OMP_NUM_THREADS=4 mpirun --oversubscribe -np 2 ./nbody_hybrid 1000 100
OMP_NUM_THREADS=2 mpirun --oversubscribe -np 4 ./nbody_hybrid 1000 100
```

---

## Step 4 — Clean Up Compiled Binaries

```bash
rm -f nbody_serial nbody_openmp nbody_pthreads nbody_mpi nbody_cuda nbody_hybrid
```

---

## Output Explained

| Output Line | Meaning |
|-------------|---------|
| `Serial Time` | Time taken by single-threaded reference run (seconds) |
| `Parallel Time` | Time taken by parallel version (seconds) |
| `Speedup` | Serial Time ÷ Parallel Time — how many times faster |
| `RMSE` | Accuracy: 0.0 = identical result to serial (perfect) |

### Speedup Formula
```
Speedup = Serial Time / Parallel Time
```
Example: Serial=0.54s, Parallel=0.18s → Speedup = 0.54/0.18 = **3.0x**

### RMSE Formula
```
RMSE = sqrt( (1/N) × Σ [ (x_serial - x_parallel)² + (y_s - y_p)² + (z_s - z_p)² ] )
```
- RMSE = **0.0** → parallel result is identical to serial ✅
- RMSE > 0 → results differ (indicates a bug)
