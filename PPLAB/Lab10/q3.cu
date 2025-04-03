#include <stdio.h>
#include <stdlib.h>

#define SECTION_SIZE 256  // Assuming a block size of 256 threads

__global__ void work_inefficient_scan_kernel(float *X, float *Y, int InputSize) {
    __shared__ float XY[SECTION_SIZE];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < InputSize) {
        XY[threadIdx.x] = X[i];
    }
    
    // The code below performs iterative scan on XY
    for (unsigned int stride = 1; stride <= threadIdx.x; stride *= 2) {
        __syncthreads();
        XY[threadIdx.x] += XY[threadIdx.x - stride];
    }
    
    if (i < InputSize) {
        Y[i] = XY[threadIdx.x];
    }
}

int main() {
    // Example input
    float h_input[] = {3, 1, 7, 0, 4, 1, 6, 3};
    int inputSize = sizeof(h_input) / sizeof(float);
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, inputSize * sizeof(float));
    cudaMalloc(&d_output, inputSize * sizeof(float));
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, inputSize * sizeof(float), cudaMemcpyHostToDevice);
    
    // Set block and grid dimensions
    int blockSize = SECTION_SIZE;
    int gridSize = (inputSize + blockSize - 1) / blockSize;
    
    // Launch kernel
    work_inefficient_scan_kernel<<<gridSize, blockSize>>>(d_input, d_output, inputSize);
    
    // Copy result back to host
    float *h_output = (float*)malloc(inputSize * sizeof(float));
    cudaMemcpy(h_output, d_output, inputSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print results
    printf("Input:  [");
    for (int i = 0; i < inputSize; i++) {
        printf("%.0f ", h_input[i]);
    }
    printf("]\n");
    
    printf("Output: [");
    for (int i = 0; i < inputSize; i++) {
        printf("%.0f ", h_output[i]);
    }
    printf("]\n");
    
    // Cleanup
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}