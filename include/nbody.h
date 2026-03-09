#ifndef NBODY_H
#define NBODY_H

#define G             6.67430e-11
#define SOFTENING     1e-9
#define DT            0.01
#define DEFAULT_N     1000
#define DEFAULT_STEPS 100
#define SEED          42

typedef struct {
    double x, y, z;
    double vx, vy, vz;
    double fx, fy, fz;
    double mass;
} Body;

void   init_bodies(Body *bodies, int n);
void   compute_forces(Body *bodies, int n);
void   update_positions(Body *bodies, int n, double dt);
double compute_rmse(const Body *a, const Body *b, int n);
double get_time_sec(void);

#endif
