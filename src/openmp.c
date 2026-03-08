/**
 * @file openmp.c
 * @brief OpenMP-parallelised N-Body simulation.
 *
 * The outer loop of the force computation is parallelised using
 * `#pragma omp parallel for`.  Because the serial version uses
 * Newton's third law (j = i+1), direct parallelisation would cause
 * race conditions on force updates.  We therefore switch to the
 * full N² approach where each thread independently computes forces
 * on its assigned bodies, eliminating data dependencies.
 *
 * Usage:  OMP_NUM_THREADS=4 ./nbody_openmp <N> <steps>
 *     or: ./nbody_openmp <N> <steps> <threads>
 */

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>


#include "nbody_common.h"
#include "nbody_io.h"
#include "nbody_types.h"


/*---------------------------------------------------------------------------
 * Force Computation — OpenMP parallel
 *---------------------------------------------------------------------------*/

/**
 * Compute gravitational forces using OpenMP.
 *
 * Each thread computes the total force on its assigned bodies independently.
 * This avoids race conditions at the cost of computing both (i,j) and (j,i)
 * interactions, but the parallelism more than compensates.
 *
 * @param bodies  Array of bodies.
 * @param n       Number of bodies.
 */
static void compute_forces_omp(Body *bodies, int n) {
#pragma omp parallel for schedule(dynamic, 64)
  for (int i = 0; i < n; i++) {
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
 * Position & Velocity Update — OpenMP parallel
 *---------------------------------------------------------------------------*/

static void update_positions_omp(Body *bodies, int n, double dt) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; i++) {
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
  int n, steps, num_threads;
  parse_args(argc, argv, &n, &steps, &num_threads);

  /* Set thread count (CLI arg overrides OMP_NUM_THREADS env var) */
  if (argc >= 4) {
    omp_set_num_threads(num_threads);
  }
  num_threads = omp_get_max_threads();

  printf("=== OpenMP N-Body Simulation ===\n");
  printf("Bodies: %d | Steps: %d | Threads: %d\n\n", n, steps, num_threads);

  Body *bodies = (Body *)malloc(n * sizeof(Body));
  if (!bodies) {
    fprintf(stderr, "Error: memory allocation failed.\n");
    return EXIT_FAILURE;
  }
  init_bodies(bodies, n, DEFAULT_SEED);

  /* Run simulation */
  double t_start = get_time_sec();

  for (int step = 0; step < steps; step++) {
    compute_forces_omp(bodies, n);
    update_positions_omp(bodies, n, DT);
  }

  double t_end = get_time_sec();
  double elapsed = t_end - t_start;

  print_timing("OpenMP", n, steps, elapsed);

  save_positions("results/openmp_output.csv", bodies, n);
  printf("Results saved to results/openmp_output.csv\n");

  free(bodies);
  return EXIT_SUCCESS;
}
