/**
 * @file verify.c
 * @brief Standalone RMSE verification tool.
 *
 * Compares each parallel output CSV against the serial baseline.
 *
 * Usage:  ./build/nbody_verify
 */

#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nbody_common.h"
#include "nbody_io.h"
#include "nbody_types.h"


typedef struct {
  const char *name;
  const char *filename;
} Paradigm;

int main(void) {
  const char *baseline_file = "results/serial_output.csv";

  Paradigm paradigms[] = {
      {"OpenMP", "results/openmp_output.csv"},
      {"Pthreads", "results/pthreads_output.csv"},
      {"MPI", "results/mpi_output.csv"},
      {"CUDA", "results/cuda_output.csv"},
      {"Hybrid", "results/hybrid_output.csv"},
  };
  int num_paradigms = sizeof(paradigms) / sizeof(paradigms[0]);

  /* Load baseline */
  int n_base;
  Body *baseline = load_positions(baseline_file, &n_base);
  if (!baseline) {
    fprintf(stderr, "Error: cannot load baseline '%s'.\n", baseline_file);
    fprintf(stderr, "Run the serial simulation first.\n");
    return EXIT_FAILURE;
  }

  printf("Baseline: %s (%d bodies)\n\n", baseline_file, n_base);
  printf("%-12s %-20s %-10s\n", "Paradigm", "RMSE", "Status");
  printf("------------------------------------------\n");

  int all_pass = 1;

  for (int p = 0; p < num_paradigms; p++) {
    int n_test;
    Body *test = load_positions(paradigms[p].filename, &n_test);

    if (!test) {
      printf("%-12s %-20s %-10s\n", paradigms[p].name, "(file not found)",
             "SKIP");
      continue;
    }

    if (n_test != n_base) {
      printf("%-12s %-20s %-10s\n", paradigms[p].name, "(size mismatch)",
             "FAIL");
      all_pass = 0;
      free(test);
      continue;
    }

    double rmse = compute_rmse(baseline, test, n_base);
    const char *status = (rmse < 1e-6) ? "PASS" : "FAIL";
    if (rmse >= 1e-6)
      all_pass = 0;

    printf("%-12s %-20.6e %-10s\n", paradigms[p].name, rmse, status);
    free(test);
  }

  printf("------------------------------------------\n");
  if (all_pass) {
    printf("\nAll paradigms PASSED accuracy verification.\n");
  } else {
    printf("\nSome paradigms FAILED. Check RMSE values above.\n");
  }

  free(baseline);
  return all_pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
