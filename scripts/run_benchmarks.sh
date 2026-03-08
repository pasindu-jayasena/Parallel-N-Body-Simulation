#!/bin/bash
# Simple benchmark script
set -e

cd "$(dirname "$0")/.."
mkdir -p results

OUT="results/benchmark.csv"
echo "paradigm,N,threads,time_sec" > "$OUT"
STEPS=100

extract() {
    grep -oP 'Time=\K[0-9.]+'
}

for N in 500 1000 2000 5000; do
    echo "--- N=$N ---"

    # Serial
    T=$(./build/nbody_serial $N $STEPS | extract)
    echo "serial,$N,1,$T" >> "$OUT"
    echo "  Serial: ${T}s"

    for THR in 1 2 4 8; do
        # OpenMP
        T=$(OMP_NUM_THREADS=$THR ./build/nbody_openmp $N $STEPS | extract)
        echo "openmp,$N,$THR,$T" >> "$OUT"
        echo "  OpenMP($THR): ${T}s"

        # Pthreads
        T=$(./build/nbody_pthreads $N $STEPS $THR | extract)
        echo "pthreads,$N,$THR,$T" >> "$OUT"
        echo "  Pthreads($THR): ${T}s"

        # MPI
        T=$(mpirun --oversubscribe -np $THR ./build/nbody_mpi $N $STEPS 2>/dev/null | extract)
        echo "mpi,$N,$THR,$T" >> "$OUT"
        echo "  MPI($THR): ${T}s"

        # Hybrid
        T=$(OMP_NUM_THREADS=$THR mpirun --oversubscribe -np 2 ./build/nbody_hybrid $N $STEPS 2>/dev/null | extract)
        echo "hybrid,$N,$THR,$T" >> "$OUT"
        echo "  Hybrid($THR): ${T}s"
    done
done

echo ""
echo "=== Benchmark Complete ==="
echo "Results in: $OUT"
echo ""
cat "$OUT"
