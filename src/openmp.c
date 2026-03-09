#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "nbody.h"

void compute_forces_omp(Body *bodies, int n) {
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        double fx = 0.0, fy = 0.0, fz = 0.0;
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            double dx = bodies[j].x - bodies[i].x;
            double dy = bodies[j].y - bodies[i].y;
            double dz = bodies[j].z - bodies[i].z;

            double dist_sq  = dx*dx + dy*dy + dz*dz + SOFTENING;
            double inv_dist = 1.0 / sqrt(dist_sq);
            double force    = G * bodies[i].mass * bodies[j].mass
                              * inv_dist * inv_dist * inv_dist;

            fx += force * dx;
            fy += force * dy;
            fz += force * dz;
        }
        bodies[i].fx = fx;
        bodies[i].fy = fy;
        bodies[i].fz = fz;
    }
}

int main(int argc, char **argv) {
    int n       = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    int steps   = (argc > 2) ? atoi(argv[2]) : DEFAULT_STEPS;
    int threads = (argc > 3) ? atoi(argv[3]) : 4;

    omp_set_num_threads(threads);

    printf("=== OpenMP N-Body Simulation ===\n");
    printf("Bodies: %d | Steps: %d | Threads: %d\n\n", n, steps, threads);

    Body *serial_bodies   = (Body *)malloc(n * sizeof(Body));
    Body *parallel_bodies = (Body *)malloc(n * sizeof(Body));

    init_bodies(serial_bodies, n);
    memcpy(parallel_bodies, serial_bodies, n * sizeof(Body));

    double t1 = get_time_sec();
    for (int s = 0; s < steps; s++) {
        compute_forces(serial_bodies, n);
        update_positions(serial_bodies, n, DT);
    }
    double serial_time = get_time_sec() - t1;

    double t2 = get_time_sec();
    for (int s = 0; s < steps; s++) {
        compute_forces_omp(parallel_bodies, n);
        update_positions(parallel_bodies, n, DT);
    }
    double parallel_time = get_time_sec() - t2;

    double rmse = compute_rmse(serial_bodies, parallel_bodies, n);

    printf("Serial Time:   %.4f seconds\n", serial_time);
    printf("Parallel Time: %.4f seconds\n", parallel_time);
    printf("Speedup:       %.2fx\n", serial_time / parallel_time);
    printf("RMSE:          %.6e\n", rmse);

    free(serial_bodies);
    free(parallel_bodies);
    return 0;
}
