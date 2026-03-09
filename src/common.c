#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "nbody.h"

void init_bodies(Body *bodies, int n) {
    srand(SEED);
    for (int i = 0; i < n; i++) {
        bodies[i].x    = ((double)rand() / RAND_MAX) * 100.0 - 50.0;
        bodies[i].y    = ((double)rand() / RAND_MAX) * 100.0 - 50.0;
        bodies[i].z    = ((double)rand() / RAND_MAX) * 100.0 - 50.0;
        bodies[i].vx   = ((double)rand() / RAND_MAX) - 0.5;
        bodies[i].vy   = ((double)rand() / RAND_MAX) - 0.5;
        bodies[i].vz   = ((double)rand() / RAND_MAX) - 0.5;
        bodies[i].fx   = 0.0;
        bodies[i].fy   = 0.0;
        bodies[i].fz   = 0.0;
        bodies[i].mass = 1.0 + ((double)rand() / RAND_MAX);
    }
}

void compute_forces(Body *bodies, int n) {
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

void update_positions(Body *bodies, int n, double dt) {
    for (int i = 0; i < n; i++) {
        bodies[i].vx += (bodies[i].fx / bodies[i].mass) * dt;
        bodies[i].vy += (bodies[i].fy / bodies[i].mass) * dt;
        bodies[i].vz += (bodies[i].fz / bodies[i].mass) * dt;
        bodies[i].x  += bodies[i].vx * dt;
        bodies[i].y  += bodies[i].vy * dt;
        bodies[i].z  += bodies[i].vz * dt;
    }
}

double compute_rmse(const Body *a, const Body *b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double dx = a[i].x - b[i].x;
        double dy = a[i].y - b[i].y;
        double dz = a[i].z - b[i].z;
        sum += dx*dx + dy*dy + dz*dz;
    }
    return sqrt(sum / n);
}

double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
