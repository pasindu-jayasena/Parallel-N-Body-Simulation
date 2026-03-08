/**
 * @file serial.c
 * @brief Serial (baseline) N-Body simulation.
 *
 * Computes all-pairs gravitational forces sequentially using Newton's
 * law of gravitation, then integrates with simple Euler method.
 *
 * Usage:  ./nbody_serial <N> <steps>
 *
 * This is the reference implementation — all parallel versions must
 * produce results matching this output within acceptable RMSE tolerance.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>


#include "nbody_common.h"
#include "nbody_io.h"
#include "nbody_types.h"


/*---------------------------------------------------------------------------
 * Force Computation — O(N²)
 *---------------------------------------------------------------------------*/

/**
 * Compute gravitational forces between all pairs of bodies.
 * Uses Newton's law: F = G * m1 * m2 / (r² + softening²)
 * Forces are accumulated symmetrically (Newton's third law) for correctness,
 * but in the serial version we compute each pair once with i < j optimisation.
 *
 * @param bodies  Array of bodies (forces are zeroed then accumulated).
 * @param n       Number of bodies.
 */
static void compute_forces(Body *bodies, int n) {
  /* Zero all forces */
  for (int i = 0; i < n; i++) {
    bodies[i].fx = 0.0;
    bodies[i].fy = 0.0;
    bodies[i].fz = 0.0;
  }

  /* All-pairs force computation with Newton's third law */
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double dx = bodies[j].x - bodies[i].x;
      double dy = bodies[j].y - bodies[i].y;
      double dz = bodies[j].z - bodies[i].z;

      double dist_sq = dx * dx + dy * dy + dz * dz + SOFTENING;
      double inv_dist = 1.0 / sqrt(dist_sq);
      double inv_dist3 = inv_dist * inv_dist * inv_dist;

      double force = G * bodies[i].mass * bodies[j].mass * inv_dist3;

      double fx = force * dx;
      double fy = force * dy;
      double fz = force * dz;

      bodies[i].fx += fx;
      bodies[i].fy += fy;
      bodies[i].fz += fz;

      bodies[j].fx -= fx;
      bodies[j].fy -= fy;
      bodies[j].fz -= fz;
    }
  }
}

/*---------------------------------------------------------------------------
 * Position & Velocity Update — Euler Integration
 *---------------------------------------------------------------------------*/

/**
 * Update velocities and positions using accumulated forces.
 *   v(t+dt) = v(t) + (F/m) * dt
 *   x(t+dt) = x(t) + v(t+dt) * dt
 *
 * @param bodies  Array of bodies.
 * @param n       Number of bodies.
 * @param dt      Timestep.
 */
static void update_positions(Body *bodies, int n, double dt) {
  for (int i = 0; i < n; i++) {
    /* Update velocity: a = F / m */
    bodies[i].vx += (bodies[i].fx / bodies[i].mass) * dt;
    bodies[i].vy += (bodies[i].fy / bodies[i].mass) * dt;
    bodies[i].vz += (bodies[i].fz / bodies[i].mass) * dt;

    /* Update position */
    bodies[i].x += bodies[i].vx * dt;
    bodies[i].y += bodies[i].vy * dt;
    bodies[i].z += bodies[i].vz * dt;
  }
}

/*---------------------------------------------------------------------------
 * Main
 *---------------------------------------------------------------------------*/

int main(int argc, char **argv) {
  int n, steps, threads;
  parse_args(argc, argv, &n, &steps, &threads);

  printf("=== Serial N-Body Simulation ===\n");
  printf("Bodies: %d | Steps: %d\n\n", n, steps);

  /* Allocate and initialise bodies */
  Body *bodies = (Body *)malloc(n * sizeof(Body));
  if (!bodies) {
    fprintf(stderr, "Error: memory allocation failed.\n");
    return EXIT_FAILURE;
  }
  init_bodies(bodies, n, DEFAULT_SEED);

  /* Run simulation */
  double t_start = get_time_sec();

  for (int step = 0; step < steps; step++) {
    compute_forces(bodies, n);
    update_positions(bodies, n, DT);
  }

  double t_end = get_time_sec();
  double elapsed = t_end - t_start;

  print_timing("Serial", n, steps, elapsed);

  /* Save final state as reference for RMSE comparisons */
  save_positions("results/serial_output.csv", bodies, n);
  printf("Results saved to results/serial_output.csv\n");

  free(bodies);
  return EXIT_SUCCESS;
}
