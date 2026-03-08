/**
 * @file nbody_types.h
 * @brief Core data types and physical constants for N-Body simulation.
 *
 * Defines the Body structure representing a particle and the fundamental
 * constants used across all simulation paradigms.
 */

#ifndef NBODY_TYPES_H
#define NBODY_TYPES_H

/*---------------------------------------------------------------------------
 * Physical Constants
 *---------------------------------------------------------------------------*/

/** Gravitational constant (scaled for simulation stability) */
#define G               6.67430e-11

/** Softening factor to prevent division-by-zero in close encounters */
#define SOFTENING       1e-9

/** Default integration timestep (seconds) */
#define DT              0.01

/** Default number of simulation steps */
#define DEFAULT_STEPS   100

/** Default number of bodies */
#define DEFAULT_N       1000

/** Default random seed for reproducibility */
#define DEFAULT_SEED    42

/*---------------------------------------------------------------------------
 * Data Structures
 *---------------------------------------------------------------------------*/

/**
 * @struct Body
 * @brief Represents a single particle/body in the simulation.
 *
 * Contains position (x,y,z), velocity (vx,vy,vz), net force (fx,fy,fz),
 * and mass. Forces are zeroed and recalculated each timestep.
 */
typedef struct {
    double x,  y,  z;      /**< Position components                      */
    double vx, vy, vz;     /**< Velocity components                      */
    double fx, fy, fz;     /**< Accumulated force components (per step)   */
    double mass;            /**< Mass of the body                         */
} Body;

#endif /* NBODY_TYPES_H */
