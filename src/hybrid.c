/**
 * @file hybrid.c
 * @brief Hybrid (MPI + OpenMP) N-Body simulation.
 *
 * Combines distributed-memory parallelism (MPI) with shared-memory
 * parallelism (OpenMP).  MPI distributes bodies across nodes/processes,
 * while OpenMP parallelises the force loop within each process.
 *
 * This model scales to multi-node clusters where each node has
 * multiple CPU cores.
 *
 * Usage:  OMP_NUM_THREADS=4 mpirun -np 2 ./nbody_hybrid <N> <steps>
 */

#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include "nbody_common.h"
#include "nbody_io.h"
#include "nbody_types.h"


/*---------------------------------------------------------------------------
 * Force Computation — MPI local + OpenMP parallel
 *---------------------------------------------------------------------------*/

/**
 * Compute forces on local bodies using OpenMP within an MPI rank.
 *
 * @param bodies   Full body array (positions up to date via Allgather).
 * @param n        Total number of bodies.
 * @param start    First body index owned by this MPI rank.
 * @param local_n  Number of bodies owned by this rank.
 */
static void compute_forces_hybrid(Body *bodies, int n, int start, int local_n) {
#pragma omp parallel for schedule(dynamic, 32)
  for (int i = start; i < start + local_n; i++) {
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (int j = 0; j < n; j++) {
      if (i == j)
        continue;

      double dx = bodies[j].x - bodies[i].x;
      double dy = bodies[j].y - bodies[i].y;
      double dz = bodies[j].z - bodies[i].z;

      double dist_sq = dx * dx + dy * dy + dz * dz + SOFTENING;
      double inv_dist = 1.0 / sqrt(dist_sq);
      double inv_dist3 = inv_dist * inv_dist * inv_dist;

      double force = G * bodies[i].mass * bodies[j].mass * inv_dist3;

      fx += force * dx;
      fy += force * dy;
      fz += force * dz;
    }

    bodies[i].fx = fx;
    bodies[i].fy = fy;
    bodies[i].fz = fz;
  }
}

/*---------------------------------------------------------------------------
 * Position & Velocity Update — local bodies, OpenMP parallel
 *---------------------------------------------------------------------------*/

static void update_positions_hybrid(Body *bodies, int start, int local_n,
                                    double dt) {
#pragma omp parallel for schedule(static)
  for (int i = start; i < start + local_n; i++) {
    bodies[i].vx += (bodies[i].fx / bodies[i].mass) * dt;
    bodies[i].vy += (bodies[i].fy / bodies[i].mass) * dt;
    bodies[i].vz += (bodies[i].fz / bodies[i].mass) * dt;

    bodies[i].x += bodies[i].vx * dt;
    bodies[i].y += bodies[i].vy * dt;
    bodies[i].z += bodies[i].vz * dt;
  }
}

/*---------------------------------------------------------------------------
 * Main
 *---------------------------------------------------------------------------*/

int main(int argc, char **argv) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

  if (provided < MPI_THREAD_FUNNELED) {
    fprintf(stderr, "Warning: MPI does not support MPI_THREAD_FUNNELED.\n");
  }

  int rank, num_procs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  int n, steps, num_threads;
  parse_args(argc, argv, &n, &steps, &num_threads);

  /* Set OpenMP threads (CLI arg or OMP_NUM_THREADS env) */
  if (argc >= 4) {
    omp_set_num_threads(num_threads);
  }
  num_threads = omp_get_max_threads();

  if (rank == 0) {
    printf("=== Hybrid (MPI + OpenMP) N-Body Simulation ===\n");
    printf("Bodies: %d | Steps: %d | MPI Procs: %d | OMP Threads/Proc: %d\n\n",
           n, steps, num_procs, num_threads);
  }

  /* All ranks keep a full copy of the body array */
  Body *bodies = (Body *)malloc(n * sizeof(Body));
  if (!bodies) {
    fprintf(stderr, "Rank %d: memory allocation failed.\n", rank);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  init_bodies(bodies, n, DEFAULT_SEED);

  /* Compute work distribution across MPI ranks */
  int chunk = n / num_procs;
  int remainder = n % num_procs;
  int local_n = chunk + (rank < remainder ? 1 : 0);
  int start = rank * chunk + (rank < remainder ? rank : remainder);

  /* Allgatherv setup */
  int *counts = (int *)malloc(num_procs * sizeof(int));
  int *displs = (int *)malloc(num_procs * sizeof(int));

  for (int r = 0; r < num_procs; r++) {
    int r_local = chunk + (r < remainder ? 1 : 0);
    int r_start = r * chunk + (r < remainder ? r : remainder);
    counts[r] = r_local * sizeof(Body);
    displs[r] = r_start * sizeof(Body);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double t_start = MPI_Wtime();

  for (int step = 0; step < steps; step++) {
    /* OpenMP-accelerated force computation on local bodies */
    compute_forces_hybrid(bodies, n, start, local_n);

    /* OpenMP-accelerated position update on local bodies */
    update_positions_hybrid(bodies, start, local_n, DT);

    /* MPI: share updated bodies across all ranks */
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_BYTE, bodies, counts, displs, MPI_BYTE,
                   MPI_COMM_WORLD);
  }

  double t_end = MPI_Wtime();
  double elapsed = t_end - t_start;

  if (rank == 0) {
    print_timing("Hybrid", n, steps, elapsed);
    save_positions("results/hybrid_output.csv", bodies, n);
    printf("Results saved to results/hybrid_output.csv\n");
  }

  free(counts);
  free(displs);
  free(bodies);

  MPI_Finalize();
  return EXIT_SUCCESS;
}
