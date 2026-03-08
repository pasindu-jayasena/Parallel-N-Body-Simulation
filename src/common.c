/**
 * @file common.c
 * @brief Implementations for nbody_common.h and nbody_io.h.
 *
 * Shared across every simulation executable so algorithm files
 * remain concise and focused on their parallelisation strategy.
 */

/* Enable POSIX APIs (clock_gettime, CLOCK_MONOTONIC) under strict C99 */
#define _POSIX_C_SOURCE 199309L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "nbody_common.h"
#include "nbody_io.h"
#include "nbody_types.h"

/*===========================================================================
 * Initialisation
 *===========================================================================*/

void init_bodies(Body *bodies, int n, unsigned int seed) {
  srand(seed);

  for (int i = 0; i < n; i++) {
    /* Positions in range [-50, 50] */
    bodies[i].x = ((double)rand() / RAND_MAX) * 100.0 - 50.0;
    bodies[i].y = ((double)rand() / RAND_MAX) * 100.0 - 50.0;
    bodies[i].z = ((double)rand() / RAND_MAX) * 100.0 - 50.0;

    /* Small initial velocities in range [-0.5, 0.5] */
    bodies[i].vx = ((double)rand() / RAND_MAX) - 0.5;
    bodies[i].vy = ((double)rand() / RAND_MAX) - 0.5;
    bodies[i].vz = ((double)rand() / RAND_MAX) - 0.5;

    /* Zero initial forces */
    bodies[i].fx = 0.0;
    bodies[i].fy = 0.0;
    bodies[i].fz = 0.0;

    /* Uniform mass in range [1.0, 2.0] */
    bodies[i].mass = 1.0 + ((double)rand() / RAND_MAX);
  }
}

/*===========================================================================
 * Accuracy Verification
 *===========================================================================*/

double compute_rmse(const Body *a, const Body *b, int n) {
  double sum = 0.0;

  for (int i = 0; i < n; i++) {
    double dx = a[i].x - b[i].x;
    double dy = a[i].y - b[i].y;
    double dz = a[i].z - b[i].z;
    sum += dx * dx + dy * dy + dz * dz;
  }

  return sqrt(sum / n);
}

/*===========================================================================
 * Timing
 *===========================================================================*/

double get_time_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/*===========================================================================
 * CSV I/O
 *===========================================================================*/

void save_positions(const char *filename, const Body *bodies, int n) {
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    fprintf(stderr, "Error: cannot open '%s' for writing.\n", filename);
    return;
  }

  fprintf(fp, "x,y,z,vx,vy,vz,mass\n");
  for (int i = 0; i < n; i++) {
    fprintf(fp, "%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e\n", bodies[i].x,
            bodies[i].y, bodies[i].z, bodies[i].vx, bodies[i].vy, bodies[i].vz,
            bodies[i].mass);
  }

  fclose(fp);
}

Body *load_positions(const char *filename, int *n) {
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Error: cannot open '%s' for reading.\n", filename);
    return NULL;
  }

  /* Count data lines (skip header) */
  char line[512];
  int count = 0;
  fgets(line, sizeof(line), fp); /* skip header */
  while (fgets(line, sizeof(line), fp)) {
    count++;
  }
  rewind(fp);
  fgets(line, sizeof(line), fp); /* skip header again */

  Body *bodies = (Body *)malloc(count * sizeof(Body));
  if (!bodies) {
    fclose(fp);
    return NULL;
  }

  for (int i = 0; i < count; i++) {
    if (fscanf(fp, "%lf,%lf,%lf,%lf,%lf,%lf,%lf", &bodies[i].x, &bodies[i].y,
               &bodies[i].z, &bodies[i].vx, &bodies[i].vy, &bodies[i].vz,
               &bodies[i].mass) != 7) {
      fprintf(stderr, "Warning: parse error at line %d in '%s'.\n", i + 2,
              filename);
    }
    bodies[i].fx = bodies[i].fy = bodies[i].fz = 0.0;
  }

  fclose(fp);
  *n = count;
  return bodies;
}

/*===========================================================================
 * CLI Helpers
 *===========================================================================*/

void parse_args(int argc, char **argv, int *n, int *steps, int *num_threads) {
  *n = DEFAULT_N;
  *steps = DEFAULT_STEPS;
  *num_threads = 1;

  if (argc >= 2)
    *n = atoi(argv[1]);
  if (argc >= 3)
    *steps = atoi(argv[2]);
  if (argc >= 4)
    *num_threads = atoi(argv[3]);

  if (*n <= 0)
    *n = DEFAULT_N;
  if (*steps <= 0)
    *steps = DEFAULT_STEPS;
  if (*num_threads <= 0)
    *num_threads = 1;
}

void print_timing(const char *paradigm, int n, int steps, double elapsed) {
  printf("[%-10s] N=%-6d Steps=%-4d Time=%.4f s\n", paradigm, n, steps,
         elapsed);
}
