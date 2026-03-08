/**
 * @file cuda_nbody.cu
 * @brief CUDA GPU-accelerated N-Body simulation.
 *
 * Each GPU thread computes the total gravitational force on one body.
 * Uses shared memory tiling to reduce global memory bandwidth pressure:
 * bodies are loaded in tiles into shared memory, and all threads in a
 * block reuse the tile for their force accumulations.
 *
 * Usage:  ./nbody_cuda <N> <steps>
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


/* ---- Include C headers (compiled as C++) ---- */
extern "C" {
#include "nbody_common.h"
#include "nbody_io.h"
#include "nbody_types.h"

}

/** Number of threads per CUDA block */
#define BLOCK_SIZE 256

/*---------------------------------------------------------------------------
 * CUDA Error Check Macro
 *---------------------------------------------------------------------------*/

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err),    \
              __FILE__, __LINE__);                                             \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/*---------------------------------------------------------------------------
 * Device Kernel — Force Computation with Shared Memory Tiling
 *---------------------------------------------------------------------------*/

/**
 * Each thread computes the net force on one body.
 * Bodies are loaded in tiles of BLOCK_SIZE into shared memory so that
 * global memory reads are coalesced and reused within the block.
 *
 * @param d_bodies  Device body array.
 * @param n         Total number of bodies.
 */
__global__ void compute_forces_kernel(Body *d_bodies, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  double px = d_bodies[i].x;
  double py = d_bodies[i].y;
  double pz = d_bodies[i].z;
  double mi = d_bodies[i].mass;

  double fx = 0.0, fy = 0.0, fz = 0.0;

  /* Shared memory tile for a block of bodies */
  __shared__ double tile_x[BLOCK_SIZE];
  __shared__ double tile_y[BLOCK_SIZE];
  __shared__ double tile_z[BLOCK_SIZE];
  __shared__ double tile_mass[BLOCK_SIZE];

  int num_tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

  for (int tile = 0; tile < num_tiles; tile++) {
    int j = tile * BLOCK_SIZE + threadIdx.x;

    /* Load tile into shared memory */
    if (j < n) {
      tile_x[threadIdx.x] = d_bodies[j].x;
      tile_y[threadIdx.x] = d_bodies[j].y;
      tile_z[threadIdx.x] = d_bodies[j].z;
      tile_mass[threadIdx.x] = d_bodies[j].mass;
    } else {
      tile_x[threadIdx.x] = 0.0;
      tile_y[threadIdx.x] = 0.0;
      tile_z[threadIdx.x] = 0.0;
      tile_mass[threadIdx.x] = 0.0;
    }
    __syncthreads();

    /* Compute interactions with all bodies in the tile */
    for (int k = 0; k < BLOCK_SIZE; k++) {
      int global_j = tile * BLOCK_SIZE + k;
      if (global_j >= n || global_j == i)
        continue;

      double dx = tile_x[k] - px;
      double dy = tile_y[k] - py;
      double dz = tile_z[k] - pz;

      double dist_sq = dx * dx + dy * dy + dz * dz + SOFTENING;
      double inv_dist = rsqrt(dist_sq); /* fast reciprocal sqrt */
      double inv_dist3 = inv_dist * inv_dist * inv_dist;

      double force = G * mi * tile_mass[k] * inv_dist3;

      fx += force * dx;
      fy += force * dy;
      fz += force * dz;
    }
    __syncthreads();
  }

  d_bodies[i].fx = fx;
  d_bodies[i].fy = fy;
  d_bodies[i].fz = fz;
}

/*---------------------------------------------------------------------------
 * Device Kernel — Position & Velocity Update
 *---------------------------------------------------------------------------*/

__global__ void update_positions_kernel(Body *d_bodies, int n, double dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  d_bodies[i].vx += (d_bodies[i].fx / d_bodies[i].mass) * dt;
  d_bodies[i].vy += (d_bodies[i].fy / d_bodies[i].mass) * dt;
  d_bodies[i].vz += (d_bodies[i].fz / d_bodies[i].mass) * dt;

  d_bodies[i].x += d_bodies[i].vx * dt;
  d_bodies[i].y += d_bodies[i].vy * dt;
  d_bodies[i].z += d_bodies[i].vz * dt;
}

/*---------------------------------------------------------------------------
 * Main
 *---------------------------------------------------------------------------*/

int main(int argc, char **argv) {
  int n, steps, dummy;
  parse_args(argc, argv, &n, &steps, &dummy);

  printf("=== CUDA N-Body Simulation ===\n");
  printf("Bodies: %d | Steps: %d | Block Size: %d\n\n", n, steps, BLOCK_SIZE);

  /* Host allocation and initialisation */
  Body *h_bodies = (Body *)malloc(n * sizeof(Body));
  if (!h_bodies) {
    fprintf(stderr, "Error: host memory allocation failed.\n");
    return EXIT_FAILURE;
  }
  init_bodies(h_bodies, n, DEFAULT_SEED);

  /* Device allocation */
  Body *d_bodies;
  CUDA_CHECK(cudaMalloc(&d_bodies, n * sizeof(Body)));
  CUDA_CHECK(
      cudaMemcpy(d_bodies, h_bodies, n * sizeof(Body), cudaMemcpyHostToDevice));

  /* Kernel launch configuration */
  int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

  /* Run simulation */
  double t_start = get_time_sec();

  for (int step = 0; step < steps; step++) {
    compute_forces_kernel<<<grid_size, BLOCK_SIZE>>>(d_bodies, n);
    update_positions_kernel<<<grid_size, BLOCK_SIZE>>>(d_bodies, n, DT);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  double t_end = get_time_sec();
  double elapsed = t_end - t_start;

  /* Copy results back to host */
  CUDA_CHECK(
      cudaMemcpy(h_bodies, d_bodies, n * sizeof(Body), cudaMemcpyDeviceToHost));

  print_timing("CUDA", n, steps, elapsed);

  save_positions("results/cuda_output.csv", h_bodies, n);
  printf("Results saved to results/cuda_output.csv\n");

  /* Cleanup */
  cudaFree(d_bodies);
  free(h_bodies);

  return EXIT_SUCCESS;
}
