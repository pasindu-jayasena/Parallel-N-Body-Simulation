#define _POSIX_C_SOURCE 200809L

#include "nbody.h"
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// shared data — all threads access these globals
static Body *g_bodies;
static int g_n, g_steps;
static pthread_barrier_t g_barrier;

// each thread gets a range [start, end) of bodies to process
typedef struct {
  int start;
  int end;
} ThreadArg;

void *thread_worker(void *arg) {
  ThreadArg *ta = (ThreadArg *)arg;

  for (int s = 0; s < g_steps; s++) {

    // phase 1: each thread computes forces for its assigned bodies
    for (int i = ta->start; i < ta->end; i++) {
      double fx = 0.0, fy = 0.0, fz = 0.0;
      for (int j = 0; j < g_n; j++) {
        if (i == j)
          continue;
        double dx = g_bodies[j].x - g_bodies[i].x;
        double dy = g_bodies[j].y - g_bodies[i].y;
        double dz = g_bodies[j].z - g_bodies[i].z;

        double dist_sq = dx * dx + dy * dy + dz * dz + SOFTENING;
        double inv_dist = 1.0 / sqrt(dist_sq);
        double force = G * g_bodies[i].mass * g_bodies[j].mass * inv_dist *
                       inv_dist * inv_dist;

        fx += force * dx;
        fy += force * dy;
        fz += force * dz;
      }
      g_bodies[i].fx = fx;
      g_bodies[i].fy = fy;
      g_bodies[i].fz = fz;
    }

    // wait for ALL threads to finish computing forces before moving bodies
    pthread_barrier_wait(&g_barrier);

    // phase 2: each thread updates positions for its assigned bodies
    for (int i = ta->start; i < ta->end; i++) {
      g_bodies[i].vx += (g_bodies[i].fx / g_bodies[i].mass) * DT;
      g_bodies[i].vy += (g_bodies[i].fy / g_bodies[i].mass) * DT;
      g_bodies[i].vz += (g_bodies[i].fz / g_bodies[i].mass) * DT;
      g_bodies[i].x += g_bodies[i].vx * DT;
      g_bodies[i].y += g_bodies[i].vy * DT;
      g_bodies[i].z += g_bodies[i].vz * DT;
    }

    // wait for ALL threads before starting next timestep
    pthread_barrier_wait(&g_barrier);
  }
  return NULL;
}

int main(int argc, char **argv) {
  int n = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
  int steps = (argc > 2) ? atoi(argv[2]) : DEFAULT_STEPS;
  int threads = (argc > 3) ? atoi(argv[3]) : 4;

  printf("=== POSIX Threads N-Body Simulation ===\n");
  printf("Bodies: %d | Steps: %d | Threads: %d\n\n", n, steps, threads);

  Body *serial_bodies = (Body *)malloc(n * sizeof(Body));
  g_bodies = (Body *)malloc(n * sizeof(Body));

  init_bodies(serial_bodies, n);
  memcpy(g_bodies, serial_bodies, n * sizeof(Body)); // identical start

  // run serial for reference
  double t1 = get_time_sec();
  for (int s = 0; s < steps; s++) {
    compute_forces(serial_bodies, n);
    update_positions(serial_bodies, n, DT);
  }
  double serial_time = get_time_sec() - t1;

  g_n = n;
  g_steps = steps;
  pthread_barrier_init(&g_barrier, NULL, threads); // barrier for 'threads' threads

  pthread_t *tids = (pthread_t *)malloc(threads * sizeof(pthread_t));
  ThreadArg *args = (ThreadArg *)malloc(threads * sizeof(ThreadArg));

  // divide bodies evenly across threads
  int chunk = n / threads;
  int remainder = n % threads;
  int offset = 0;

  double t2 = get_time_sec();
  for (int t = 0; t < threads; t++) {
    args[t].start = offset;
    args[t].end = offset + chunk + (t < remainder ? 1 : 0);
    offset = args[t].end;
    pthread_create(&tids[t], NULL, thread_worker, &args[t]); // launch thread
  }
  for (int t = 0; t < threads; t++) {
    pthread_join(tids[t], NULL); // wait for thread to finish
  }
  double parallel_time = get_time_sec() - t2;

  double rmse = compute_rmse(serial_bodies, g_bodies, n);

  printf("Serial Time:   %.4f seconds\n", serial_time);
  printf("Parallel Time: %.4f seconds\n", parallel_time);
  printf("Speedup:       %.2fx\n", serial_time / parallel_time);
  printf("RMSE:          %.6e\n", rmse);

  pthread_barrier_destroy(&g_barrier);
  free(tids);
  free(args);
  free(serial_bodies);
  free(g_bodies);
  return 0;
}