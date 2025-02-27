#include <stdio.h>
#include <string.h>
#include <cuda.h>

// Kernel to generate output string T
__global__ void generateOutputString(const char *Sin, char *T, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        int start = idx * (idx + 1) / 2; // Start index in T for current character
        for (int i = 0; i <= idx; i++) {
            T[start + i] = Sin[idx];
        }
    }
}

int main() {
    char Sin[] = "Hai";
    int len = strlen(Sin);
    int T_len = len * (len + 1) / 2; // Length of output string T

    // Allocate host and device memory
    char h_T[T_len + 1];
    char *d_Sin, *d_T;
    cudaMalloc((void **)&d_Sin, len + 1);
    cudaMalloc((void **)&d_T, T_len + 1);

    // Copy input string to device
    cudaMemcpy(d_Sin, Sin, len + 1, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
    generateOutputString<<<blocksPerGrid, threadsPerBlock>>>(d_Sin, d_T, len);

    // Copy result back to host
    cudaMemcpy(h_T, d_T, T_len + 1, cudaMemcpyDeviceToHost);
    h_T[T_len] = '\0'; // Null-terminate the string

    // Print the result
    printf("Input: %s\n", Sin);
    printf("Output: %s\n", h_T);

    // Free device memory
    cudaFree(d_Sin);
    cudaFree(d_T);

    return 0;
}