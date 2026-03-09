#include "nbody.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void compute_forces_local(Body *bodies, int n, int start, int local_n) {
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
      double force =
          G * bodies[i].mass * bodies[j].mass * inv_dist * inv_dist * inv_dist;

      fx += force * dx;
      fy += force * dy;
      fz += force * dz;
    }
    bodies[i].fx = fx;
    bodies[i].fy = fy;
    bodies[i].fz = fz;
  }
}

void update_local(Body *bodies, int start, int local_n) {
  for (int i = start; i < start + local_n; i++) {
    bodies[i].vx += (bodies[i].fx / bodies[i].mass) * DT;
    bodies[i].vy += (bodies[i].fy / bodies[i].mass) * DT;
    bodies[i].vz += (bodies[i].fz / bodies[i].mass) * DT;
    bodies[i].x += bodies[i].vx * DT;
    bodies[i].y += bodies[i].vy * DT;
    bodies[i].z += bodies[i].vz * DT;
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
  int steps = (argc > 2) ? atoi(argv[2]) : DEFAULT_STEPS;

  if (rank == 0) {
    printf("=== MPI N-Body Simulation ===\n");
    printf("Bodies: %d | Steps: %d | Processes: %d\n\n", n, steps, size);
  }

  Body *bodies = (Body *)malloc(n * sizeof(Body));
  init_bodies(bodies, n);

  Body *serial_ref = NULL;
  double serial_time = 0.0;
  if (rank == 0) {
    serial_ref = (Body *)malloc(n * sizeof(Body));
    memcpy(serial_ref, bodies, n * sizeof(Body));

    double t1 = get_time_sec();
    for (int s = 0; s < steps; s++) {
      compute_forces(serial_ref, n);
      update_positions(serial_ref, n, DT);
    }
    serial_time = get_time_sec() - t1;
  }

  int chunk = n / size;
  int remainder = n % size;
  int local_n = chunk + (rank < remainder ? 1 : 0);
  int start = rank * chunk + (rank < remainder ? rank : remainder);

  int *counts = (int *)malloc(size * sizeof(int));
  int *displs = (int *)malloc(size * sizeof(int));
  for (int r = 0; r < size; r++) {
    int r_n = chunk + (r < remainder ? 1 : 0);
    int r_s = r * chunk + (r < remainder ? r : remainder);
    counts[r] = r_n * (int)sizeof(Body);
    displs[r] = r_s * (int)sizeof(Body);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double t2 = MPI_Wtime();

  for (int s = 0; s < steps; s++) {
    compute_forces_local(bodies, n, start, local_n);
    update_local(bodies, start, local_n);

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_BYTE, bodies, counts, displs, MPI_BYTE,
                   MPI_COMM_WORLD);
  }

  double parallel_time = MPI_Wtime() - t2;

  if (rank == 0) {
    double rmse = compute_rmse(serial_ref, bodies, n);

    printf("Serial Time:   %.4f seconds\n", serial_time);
    printf("Parallel Time: %.4f seconds\n", parallel_time);
    printf("Speedup:       %.2fx\n", serial_time / parallel_time);
    printf("RMSE:          %.6e\n", rmse);

    free(serial_ref);
  }

  free(counts);
  free(displs);
  free(bodies);
  MPI_Finalize();
  return 0;
}
//