/**
 * @file pthreads.c
 * @brief POSIX Threads N-Body simulation.
 *
 * Manually partitions bodies among threads.  Each thread computes
 * forces for its assigned range, then a barrier synchronises before
 * position updates.  Demonstrates explicit thread management as an
 * alternative to the implicit model used by OpenMP.
 *
 * Usage:  ./nbody_pthreads <N> <steps> <threads>
 */

/* Enable pthread_barrier_t and POSIX timer APIs under strict C99 */
#define _POSIX_C_SOURCE 200809L
#define _XOPEN_SOURCE 600

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#include "nbody_common.h"
#include "nbody_io.h"
#include "nbody_types.h"

/*---------------------------------------------------------------------------
 * Shared state visible to all threads
 *---------------------------------------------------------------------------*/

static Body *g_bodies;
static int g_n;
static int g_steps;
static double g_dt;

static pthread_barrier_t barrier;

/*---------------------------------------------------------------------------
 * Per-thread argument
 *---------------------------------------------------------------------------*/

typedef struct {
  int thread_id;
  int start; /**< First body index (inclusive) */
  int end;   /**< Last body index  (exclusive) */
} ThreadArg;

/*---------------------------------------------------------------------------
 * Thread Worker
 *---------------------------------------------------------------------------*/

/**
 * Each thread loops over all timesteps.  Within each step:
 *   1. Compute forces for bodies [start, end)
 *   2. Barrier — wait for all threads to finish forces
 *   3. Update positions for bodies [start, end)
 *   4. Barrier — wait before next step
 *
 * @param arg  Pointer to ThreadArg.
 * @return     NULL
 */
static void *thread_worker(void *arg) {
  ThreadArg *ta = (ThreadArg *)arg;
  int start = ta->start;
  int end = ta->end;

  for (int step = 0; step < g_steps; step++) {

    /* --- Phase 1: Compute forces for assigned bodies --- */
    for (int i = start; i < end; i++) {
      double fx = 0.0, fy = 0.0, fz = 0.0;

      for (int j = 0; j < g_n; j++) {
        if (i == j)
          continue;

        double dx = g_bodies[j].x - g_bodies[i].x;
        double dy = g_bodies[j].y - g_bodies[i].y;
        double dz = g_bodies[j].z - g_bodies[i].z;

        double dist_sq = dx * dx + dy * dy + dz * dz + SOFTENING;
        double inv_dist = 1.0 / sqrt(dist_sq);
        double inv_dist3 = inv_dist * inv_dist * inv_dist;

        double force = G * g_bodies[i].mass * g_bodies[j].mass * inv_dist3;

        fx += force * dx;
        fy += force * dy;
        fz += force * dz;
      }

      g_bodies[i].fx = fx;
      g_bodies[i].fy = fy;
      g_bodies[i].fz = fz;
    }

    /* Barrier: all forces computed before updating positions */
    pthread_barrier_wait(&barrier);

    /* --- Phase 2: Update positions for assigned bodies --- */
    for (int i = start; i < end; i++) {
      g_bodies[i].vx += (g_bodies[i].fx / g_bodies[i].mass) * g_dt;
      g_bodies[i].vy += (g_bodies[i].fy / g_bodies[i].mass) * g_dt;
      g_bodies[i].vz += (g_bodies[i].fz / g_bodies[i].mass) * g_dt;

      g_bodies[i].x += g_bodies[i].vx * g_dt;
      g_bodies[i].y += g_bodies[i].vy * g_dt;
      g_bodies[i].z += g_bodies[i].vz * g_dt;
    }

    /* Barrier: all positions updated before next force computation */
    pthread_barrier_wait(&barrier);
  }

  return NULL;
}

/*---------------------------------------------------------------------------
 * Main
 *---------------------------------------------------------------------------*/

int main(int argc, char **argv) {
  int n, steps, num_threads;
  parse_args(argc, argv, &n, &steps, &num_threads);

  printf("=== POSIX Threads N-Body Simulation ===\n");
  printf("Bodies: %d | Steps: %d | Threads: %d\n\n", n, steps, num_threads);

  /* Allocate and initialise */
  g_bodies = (Body *)malloc(n * sizeof(Body));
  if (!g_bodies) {
    fprintf(stderr, "Error: memory allocation failed.\n");
    return EXIT_FAILURE;
  }
  init_bodies(g_bodies, n, DEFAULT_SEED);

  g_n = n;
  g_steps = steps;
  g_dt = DT;

  /* Initialise barrier */
  pthread_barrier_init(&barrier, NULL, num_threads);

  /* Create thread arguments with balanced work partitioning */
  pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  ThreadArg *args = (ThreadArg *)malloc(num_threads * sizeof(ThreadArg));

  int chunk = n / num_threads;
  int remainder = n % num_threads;

  double t_start = get_time_sec();

  int offset = 0;
  for (int t = 0; t < num_threads; t++) {
    args[t].thread_id = t;
    args[t].start = offset;
    args[t].end = offset + chunk + (t < remainder ? 1 : 0);
    offset = args[t].end;

    pthread_create(&threads[t], NULL, thread_worker, &args[t]);
  }

  /* Wait for all threads to complete */
  for (int t = 0; t < num_threads; t++) {
    pthread_join(threads[t], NULL);
  }

  double t_end = get_time_sec();
  double elapsed = t_end - t_start;

  print_timing("Pthreads", n, steps, elapsed);

  save_positions("results/pthreads_output.csv", g_bodies, n);
  printf("Results saved to results/pthreads_output.csv\n");

  /* Cleanup */
  pthread_barrier_destroy(&barrier);
  free(threads);
  free(args);
  free(g_bodies);

  return EXIT_SUCCESS;
}
