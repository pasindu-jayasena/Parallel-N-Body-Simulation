/**
 * @file mpi.c
 * @brief MPI distributed-memory N-Body simulation.
 *
 * Each MPI rank owns a subset of bodies and computes forces on them.
 * At each timestep, all ranks exchange updated positions via
 * MPI_Allgather so every rank has the full position state needed
 * for the O(N²) force calculation.
 *
 * Usage:  mpirun -np 4 ./nbody_mpi <N> <steps>
 */

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include "nbody_common.h"
#include "nbody_io.h"
#include "nbody_types.h"


/*---------------------------------------------------------------------------
 * Force Computation — local bodies only
 *---------------------------------------------------------------------------*/

/**
 * Compute forces for bodies in range [start, start+local_n).
 * Needs the full bodies array for interaction pairs.
 *
 * @param bodies   Full body array (positions up to date).
 * @param n        Total number of bodies.
 * @param start    First body index owned by this rank.
 * @param local_n  Number of bodies owned by this rank.
 */
static void compute_forces_local(Body *bodies, int n, int start, int local_n) {
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
 * Position & Velocity Update — local bodies only
 *---------------------------------------------------------------------------*/

static void update_positions_local(Body *bodies, int start, int local_n,
                                   double dt) {
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
  MPI_Init(&argc, &argv);

  int rank, num_procs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  int n, steps, dummy;
  parse_args(argc, argv, &n, &steps, &dummy);

  if (rank == 0) {
    printf("=== MPI N-Body Simulation ===\n");
    printf("Bodies: %d | Steps: %d | Processes: %d\n\n", n, steps, num_procs);
  }

  /* All ranks allocate the full body array (needed for force computation) */
  Body *bodies = (Body *)malloc(n * sizeof(Body));
  if (!bodies) {
    fprintf(stderr, "Rank %d: memory allocation failed.\n", rank);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  init_bodies(bodies, n, DEFAULT_SEED);

  /* Compute work distribution */
  int chunk = n / num_procs;
  int remainder = n % num_procs;
  int local_n = chunk + (rank < remainder ? 1 : 0);
  int start = rank * chunk + (rank < remainder ? rank : remainder);

  /*
   * For MPI_Allgatherv we need displacements and counts in terms of
   * doubles (each Body has 10 doubles, but we only exchange the
   * position + velocity + mass = 7 relevant fields).
   * For simplicity, we gather the entire Body array as raw bytes.
   */
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
    /* Each rank computes forces for its local bodies */
    compute_forces_local(bodies, n, start, local_n);

    /* Update local positions */
    update_positions_local(bodies, start, local_n, DT);

    /* Share updated bodies across all ranks */
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_BYTE, bodies, counts, displs, MPI_BYTE,
                   MPI_COMM_WORLD);
  }

  double t_end = MPI_Wtime();
  double elapsed = t_end - t_start;

  if (rank == 0) {
    print_timing("MPI", n, steps, elapsed);
    save_positions("results/mpi_output.csv", bodies, n);
    printf("Results saved to results/mpi_output.csv\n");
  }

  free(counts);
  free(displs);
  free(bodies);

  MPI_Finalize();
  return EXIT_SUCCESS;
}
