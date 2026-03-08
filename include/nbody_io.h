/**
 * @file nbody_io.h
 * @brief File I/O routines for saving and loading body state.
 *
 * Provides CSV-based persistence so that any simulation paradigm can
 * export its final state for comparison / RMSE verification.
 */

#ifndef NBODY_IO_H
#define NBODY_IO_H

#include "nbody_types.h"

/**
 * Save body positions (and velocities) to a CSV file.
 *
 * Format per line:  x,y,z,vx,vy,vz,mass
 *
 * @param filename  Output file path.
 * @param bodies    Array of bodies.
 * @param n         Number of bodies.
 */
void save_positions(const char *filename, const Body *bodies, int n);

/**
 * Load body state from a CSV file written by save_positions().
 *
 * @param filename  Input file path.
 * @param n         [out] Number of bodies read.
 * @return Heap-allocated array of Body; caller must free().
 *         Returns NULL on failure.
 */
Body *load_positions(const char *filename, int *n);

#endif /* NBODY_IO_H */
