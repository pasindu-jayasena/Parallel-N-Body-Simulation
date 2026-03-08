/**
 * @file nbody_common.h
 * @brief Shared utilities: initialisation, RMSE, timing, and CLI parsing.
 *
 * Every simulation executable links against common.c, which implements
 * these helpers so the core algorithm code stays clean and focused.
 */

#ifndef NBODY_COMMON_H
#define NBODY_COMMON_H

#include "nbody_types.h"

/*---------------------------------------------------------------------------
 * Initialisation
 *---------------------------------------------------------------------------*/

/**
 * Initialise bodies with deterministic pseudo-random positions, velocities,
 * and uniform mass.  Using the same seed guarantees identical initial
 * conditions across all paradigms.
 *
 * @param bodies  Pre-allocated array of bodies.
 * @param n       Number of bodies.
 * @param seed    Random seed for reproducibility.
 */
void init_bodies(Body *bodies, int n, unsigned int seed);

/*---------------------------------------------------------------------------
 * Accuracy Verification
 *---------------------------------------------------------------------------*/

/**
 * Compute Root Mean Square Error of positions between two body arrays.
 *
 *   RMSE = sqrt( (1/N) * SUM_i [ (xa-xb)^2 + (ya-yb)^2 + (za-zb)^2 ] )
 *
 * @param a  Reference body array (e.g., serial output).
 * @param b  Test body array (e.g., parallel output).
 * @param n  Number of bodies (must be the same for both).
 * @return   RMSE value; 0.0 indicates perfect agreement.
 */
double compute_rmse(const Body *a, const Body *b, int n);

/*---------------------------------------------------------------------------
 * Timing
 *---------------------------------------------------------------------------*/

/**
 * Portable wall-clock timer in seconds.
 * Uses clock_gettime(CLOCK_MONOTONIC) on POSIX systems.
 *
 * @return Current wall-clock time in seconds (arbitrary epoch).
 */
double get_time_sec(void);

/*---------------------------------------------------------------------------
 * CLI Helpers
 *---------------------------------------------------------------------------*/

/**
 * Parse standard command-line arguments: N, steps, [threads].
 *
 * Usage:  ./nbody_xxx  <N>  <steps>  [threads]
 *
 * @param argc        Argument count from main().
 * @param argv        Argument vector from main().
 * @param n           [out] Number of bodies.
 * @param steps       [out] Number of simulation steps.
 * @param num_threads [out] Thread count (defaults to 1 if not supplied).
 */
void parse_args(int argc, char **argv, int *n, int *steps, int *num_threads);

/**
 * Print a formatted timing result line to stdout.
 *
 * @param paradigm  Name of the paradigm (e.g., "Serial", "OpenMP").
 * @param n         Number of bodies.
 * @param steps     Number of timesteps.
 * @param elapsed   Elapsed wall-clock time in seconds.
 */
void print_timing(const char *paradigm, int n, int steps, double elapsed);

#endif /* NBODY_COMMON_H */
