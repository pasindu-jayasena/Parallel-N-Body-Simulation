#include <stdio.h>
#include <stdlib.h>
#include "nbody.h"

int main(int argc, char **argv) {
    int n     = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    int steps = (argc > 2) ? atoi(argv[2]) : DEFAULT_STEPS;

    printf("=== Serial N-Body Simulation ===\n");
    printf("Bodies: %d | Steps: %d\n\n", n, steps);

    Body *bodies = (Body *)malloc(n * sizeof(Body));
    init_bodies(bodies, n);

    double start = get_time_sec();
    for (int s = 0; s < steps; s++) {
        compute_forces(bodies, n);
        update_positions(bodies, n, DT);
    }
    double elapsed = get_time_sec() - start;

    printf("Serial Time: %.4f seconds\n", elapsed);

    free(bodies);
    return 0;
}
