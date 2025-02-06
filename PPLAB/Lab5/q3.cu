#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void computeSine(float *angles, float *sine_values, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) {
        sine_values[index] = sinf(angles[index]);  // Using sinf(x) for single precision
    }
}

int main() {
    int N = 1024; // Array size
    float *angles, *sine_values;
    float *d_angles, *d_sine_values;

    size_t size = N * sizeof(float);

    // Allocate memory for host arrays
    angles = (float*)malloc(size);
    sine_values = (float*)malloc(size);

    // Initialize angles array with random values (in radians)
    for (int i = 0; i < N; i++) {
        angles[i] = (float)(i * 3.14159 / 180); // Convert degrees to radians
    }

    // Allocate memory for device arrays
    cudaMalloc((void**)&d_angles, size);
    cudaMalloc((void**)&d_sine_values, size);

    // Copy data from host to device
    cudaMemcpy(d_angles, angles, size, cudaMemcpyHostToDevice);

    // Launch kernel with 256 threads per block
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    computeSine<<<blocks, threadsPerBlock>>>(d_angles, d_sine_values, N);

    // Copy result from device to host
    cudaMemcpy(sine_values, d_sine_values, size, cudaMemcpyDeviceToHost);

    // Print some sine values
    for (int i = 0; i < N; i++) {
        printf("sin(%f) = %f\n", angles[i], sine_values[i]);
    }

    // Free memory
    free(angles);
    free(sine_values);
    cudaFree(d_angles);
    cudaFree(d_sine_values);

    return 0;
}
