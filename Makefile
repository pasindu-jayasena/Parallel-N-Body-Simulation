################################################################################
# Makefile — Parallel N-Body Simulation
#
# Builds all paradigms:  serial, openmp, pthreads, mpi, cuda, hybrid
#
# Usage:
#   make all           Build everything (excluding CUDA if nvcc unavailable)
#   make serial        Build serial baseline
#   make openmp        Build OpenMP version
#   make pthreads      Build POSIX Threads version
#   make mpi           Build MPI version
#   make cuda          Build CUDA version (requires nvcc)
#   make hybrid        Build Hybrid MPI+OpenMP version
#   make clean         Remove all build artefacts
#
# Environment:
#   Requires GCC (or compatible), with OpenMP support.
#   MPI targets require mpicc to be in PATH.
#   CUDA target requires nvcc to be in PATH.
################################################################################

# ---- Compiler Configuration ------------------------------------------------

CC        = gcc
MPICC     = mpicc
NVCC      = nvcc

CFLAGS    = -O2 -Wall -Wextra -std=c99
LDFLAGS   = -lm

INC       = -Iinclude
SRC       = src

# Output directory
BUILD     = build

# ---- Source Files -----------------------------------------------------------

COMMON_SRC = $(SRC)/common.c

# ---- Targets ----------------------------------------------------------------

.PHONY: all serial openmp pthreads mpi cuda hybrid clean dirs

all: dirs serial openmp pthreads mpi hybrid
	@echo ""
	@echo "=== All targets built successfully ==="
	@echo "  Binaries are in $(BUILD)/"
	@echo "  Run 'make cuda' separately if nvcc is available."

dirs:
	@mkdir -p $(BUILD) results

# ---- Serial -----------------------------------------------------------------

serial: dirs
	$(CC) $(CFLAGS) $(INC) -o $(BUILD)/nbody_serial \
		$(SRC)/serial.c $(COMMON_SRC) $(LDFLAGS)
	@echo "[OK] nbody_serial"

# ---- OpenMP -----------------------------------------------------------------

openmp: dirs
	$(CC) $(CFLAGS) $(INC) -fopenmp -o $(BUILD)/nbody_openmp \
		$(SRC)/openmp.c $(COMMON_SRC) $(LDFLAGS)
	@echo "[OK] nbody_openmp"

# ---- POSIX Threads ----------------------------------------------------------

pthreads: dirs
	$(CC) $(CFLAGS) $(INC) -pthread -o $(BUILD)/nbody_pthreads \
		$(SRC)/pthreads.c $(COMMON_SRC) $(LDFLAGS)
	@echo "[OK] nbody_pthreads"

# ---- MPI --------------------------------------------------------------------

mpi: dirs
	$(MPICC) $(CFLAGS) $(INC) -o $(BUILD)/nbody_mpi \
		$(SRC)/mpi.c $(COMMON_SRC) $(LDFLAGS)
	@echo "[OK] nbody_mpi"

# ---- CUDA -------------------------------------------------------------------

cuda: dirs
	$(NVCC) -O2 $(INC) -o $(BUILD)/nbody_cuda \
		$(SRC)/cuda_nbody.cu $(COMMON_SRC) $(LDFLAGS)
	@echo "[OK] nbody_cuda"

# ---- Hybrid (MPI + OpenMP) --------------------------------------------------

hybrid: dirs
	$(MPICC) $(CFLAGS) $(INC) -fopenmp -o $(BUILD)/nbody_hybrid \
		$(SRC)/hybrid.c $(COMMON_SRC) $(LDFLAGS)
	@echo "[OK] nbody_hybrid"

# ---- Clean ------------------------------------------------------------------

clean:
	rm -rf $(BUILD)
	@echo "Build directory cleaned."
