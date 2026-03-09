#include "nbody.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLOCK_SIZE 256 // threads per GPU block

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));            \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// GPU kernel: one thread per body, uses shared memory to reduce global reads
__global__ void compute_forces_kernel(Body *bodies, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  double px = bodies[i].x, py = bodies[i].y, pz = bodies[i].z;
  double mi = bodies[i].mass;
  double fx = 0.0, fy = 0.0, fz = 0.0;

  // shared memory holds a tile of bodies — much faster than global memory
  __shared__ double sx[BLOCK_SIZE], sy[BLOCK_SIZE];
  __shared__ double sz[BLOCK_SIZE], sm[BLOCK_SIZE];

  int tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  for (int t = 0; t < tiles; t++) {
    // each thread loads one body into shared memory
    int j = t * BLOCK_SIZE + threadIdx.x;
    if (j < n) {
      sx[threadIdx.x] = bodies[j].x;
      sy[threadIdx.x] = bodies[j].y;
      sz[threadIdx.x] = bodies[j].z;
      sm[threadIdx.x] = bodies[j].mass;
    } else {
      sx[threadIdx.x] = sy[threadIdx.x] = sz[threadIdx.x] = 0.0;
      sm[threadIdx.x] = 0.0;
    }
    __syncthreads(); // wait until all threads have loaded the tile

    // compute forces against all bodies in this tile
    for (int k = 0; k < BLOCK_SIZE; k++) {
      int gj = t * BLOCK_SIZE + k;
      if (gj >= n || gj == i)
        continue;

      double dx = sx[k] - px, dy = sy[k] - py, dz = sz[k] - pz;
      double dist_sq = dx * dx + dy * dy + dz * dz + SOFTENING;
      double inv_dist = rsqrt(dist_sq); // fast GPU inverse sqrt
      double force = G * mi * sm[k] * inv_dist * inv_dist * inv_dist;

      fx += force * dx;
      fy += force * dy;
      fz += force * dz;
    }
    __syncthreads(); // done with this tile before loading next
  }

  bodies[i].fx = fx;
  bodies[i].fy = fy;
  bodies[i].fz = fz;
}

// GPU kernel: update position for each body
__global__ void update_kernel(Body *bodies, int n, double dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  bodies[i].vx += (bodies[i].fx / bodies[i].mass) * dt;
  bodies[i].vy += (bodies[i].fy / bodies[i].mass) * dt;
  bodies[i].vz += (bodies[i].fz / bodies[i].mass) * dt;
  bodies[i].x += bodies[i].vx * dt;
  bodies[i].y += bodies[i].vy * dt;
  bodies[i].z += bodies[i].vz * dt;
}

int main(int argc, char **argv) {
  int n = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
  int steps = (argc > 2) ? atoi(argv[2]) : DEFAULT_STEPS;

  printf("=== CUDA N-Body Simulation ===\n");
  printf("Bodies: %d | Steps: %d\n\n", n, steps);

  // CPU memory: serial reference + host copy for GPU transfer
  Body *serial_bodies = (Body *)malloc(n * sizeof(Body));
  Body *h_bodies = (Body *)malloc(n * sizeof(Body));
  init_bodies(serial_bodies, n);
  memcpy(h_bodies, serial_bodies, n * sizeof(Body));

  // run serial on CPU
  double t1 = get_time_sec();
  for (int s = 0; s < steps; s++) {
    compute_forces(serial_bodies, n);
    update_positions(serial_bodies, n, DT);
  }
  double serial_time = get_time_sec() - t1;

  // GPU memory: allocate and copy data from CPU to GPU
  Body *d_bodies;
  CUDA_CHECK(cudaMalloc(&d_bodies, n * sizeof(Body)));
  CUDA_CHECK(
      cudaMemcpy(d_bodies, h_bodies, n * sizeof(Body), cudaMemcpyHostToDevice));

  int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE; // number of blocks needed

  // run simulation on GPU
  double t2 = get_time_sec();
  for (int s = 0; s < steps; s++) {
    compute_forces_kernel<<<grid, BLOCK_SIZE>>>(d_bodies, n);
    update_kernel<<<grid, BLOCK_SIZE>>>(d_bodies, n, DT);
    cudaDeviceSynchronize(); // wait for GPU before next step
  }
  double cuda_time = get_time_sec() - t2;

  // copy results back from GPU to CPU
  CUDA_CHECK(
      cudaMemcpy(h_bodies, d_bodies, n * sizeof(Body), cudaMemcpyDeviceToHost));

  double rmse = compute_rmse(serial_bodies, h_bodies, n);

  printf("Serial Time: %.4f seconds\n", serial_time);
  printf("CUDA Time:   %.4f seconds\n", cuda_time);
  printf("Speedup:     %.2fx\n", serial_time / cuda_time);
  printf("RMSE:        %.6e\n", rmse);

  cudaFree(d_bodies);
  free(serial_bodies);
  free(h_bodies);
  return 0;
}