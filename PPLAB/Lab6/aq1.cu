#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void onesComplementKernel(int *input, int *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Ensure we are within bounds
    if (idx < N) {
        // Perform one's complement (flip the bits)
        output[idx] = ~input[idx];
    }
}

int main() {
    int N = 10; // Size of the input array
    int *h_input = (int *)malloc(N * sizeof(int));    // Host input array
    int *h_output = (int *)malloc(N * sizeof(int));   // Host output array
    int *d_input, *d_output;

    // Initialize the input array with binary numbers (e.g., 5 -> 101)
    for (int i = 0; i < N; i++) {
        h_input[i] = rand() % 1024;  // Random binary numbers (between 0 and 1023)
    }

    // Allocate memory on the device
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, N * sizeof(int));

    // Copy the input array from host to device
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with appropriate number of threads and blocks
    int blockSize = 256;  // Set block size to 256 threads
    int numBlocks = (N + blockSize - 1) / blockSize;  // Calculate number of blocks needed

    onesComplementKernel<<<numBlocks, blockSize>>>(d_input, d_output, N);

    // Check for errors during kernel execution
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    // Copy the result array from device to host
    cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the input and output arrays
    printf("Input Binary Numbers: \n");
    for (int i = 0; i < N; i++) {
        printf("%d: %d\n", i, h_input[i]);
    }

    printf("\nOne's Complement of Binary Numbers: \n");
    for (int i = 0; i < N; i++) {
        printf("%d: %d\n", i, h_output[i]);
    }

    // Free device and host memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
