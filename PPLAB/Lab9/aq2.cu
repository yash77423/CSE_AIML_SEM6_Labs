#include <stdio.h>
#include <cuda.h>
#include <string.h>

#define M 2
#define N 4

__global__ void repeat_chars_kernel(char *A, int *B, char *STR, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int row = idx / cols;
        int col = idx % cols;
        int repeat = B[row * cols + col]; // Number of repetitions
        int offset = 0;

        // Calculate the starting position in STR for this thread
        for (int i = 0; i < row * cols + col; i++) {
            offset += B[i];
        }

        // Fill STR with the repeated character
        for (int i = 0; i < repeat; i++) {
            STR[offset + i] = A[row * cols + col];
        }
    }
}

int main() {
    char A[M][N] = {{'P', 'C', 'a', 'P'}, {'e', 'X', 'a', 'M'}};
    int B[M][N] = {{1, 2, 4, 3}, {2, 4, 3, 2}};
    char STR[100] = {0}; // Output string
    char *d_A, *d_STR;
    int *d_B;

    // Calculate total length of STR
    int total_len = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            total_len += B[i][j];
        }
    }

    // Allocate device memory
    cudaMalloc((void **)&d_A, M * N * sizeof(char));
    cudaMalloc((void **)&d_B, M * N * sizeof(int));
    cudaMalloc((void **)&d_STR, total_len * sizeof(char));

    // Copy data to device
    cudaMemcpy(d_A, A, M * N * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, M * N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    repeat_chars_kernel<<<1, M * N>>>(d_A, d_B, d_STR, M, N);

    // Copy result back to host
    cudaMemcpy(STR, d_STR, total_len * sizeof(char), cudaMemcpyDeviceToHost);

    // Print result
    printf("Output string STR: %s\n", STR);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_STR);

    return 0;
}