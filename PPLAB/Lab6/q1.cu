#include <stdio.h>
#include <cuda_runtime.h>

__global__ void convolution_1D_basic_kernel(float *N, float *M, float *P, int Mask_Width, int Width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  
    // Initialize the output value for this thread
    float Pvalue = 0.0f;
  
    // Calculate the starting point in the input array
    int N_start_point = i - (Mask_Width / 2);
  
    // Loop through the mask elements and accumulate the result
    for (int j = 0; j < Mask_Width; j++) {
        // Ensure the index is within the bounds of the input array
        if (N_start_point + j >= 0 && N_start_point + j < Width) {
            Pvalue += N[N_start_point + j] * M[j];
        }
    }

    // Store the result
    if (i < Width) {
        P[i] = Pvalue;
    }
}

int main() {
    int Width = 1024;         // Example size of input array
    int Mask_Width = 5;       // Example mask size

    // Allocate memory for arrays
    float *N, *M, *P;
    float *d_N, *d_M, *d_P;

    N = (float*)malloc(Width * sizeof(float));
    M = (float*)malloc(Mask_Width * sizeof(float));
    P = (float*)malloc(Width * sizeof(float));

    // Initialize input array N and mask M with some values (example)
    for (int i = 0; i < Width; i++) {
        N[i] = (float)(i + 1);
    }
    for (int i = 0; i < Mask_Width; i++) {
        M[i] = (float)(i + 1);
    }

    // Allocate device memory
    cudaMalloc((void**)&d_N, Width * sizeof(float));
    cudaMalloc((void**)&d_M, Mask_Width * sizeof(float));
    cudaMalloc((void**)&d_P, Width * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_N, N, Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, M, Mask_Width * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch parameters
    int blockSize = 256;  // Number of threads per block
    int gridSize = (Width + blockSize - 1) / blockSize;

    // Launch kernel
    convolution_1D_basic_kernel<<<gridSize, blockSize>>>(d_N, d_M, d_P, Mask_Width, Width);

    // Copy the result back to host
    cudaMemcpy(P, d_P, Width * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result (for checking purposes)
    for (int i = 0; i < Width; i++) {
        printf("%f ", P[i]);
    }
    printf("\n");

    // Free memory
    free(N);
    free(M);
    free(P);
    cudaFree(d_N);
    cudaFree(d_M);
    cudaFree(d_P);

    return 0;
}
