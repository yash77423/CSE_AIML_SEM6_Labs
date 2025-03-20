#include <stdio.h>
#include <cuda.h>

#define M 4
#define N 4

__global__ void complement_kernel(int *A, int *B, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        if (row > 0 && row < rows - 1 && col > 0 && col < cols - 1) {
            B[row * cols + col] = ~A[row * cols + col]; // 1's complement
        } else {
            B[row * cols + col] = A[row * cols + col]; // Border remains same
        }
    }
}

int main() {
    int A[M][N] = {{1, 2, 3, 4}, {6, 5, 8, 3}, {2, 4, 10, 1}, {9, 1, 2, 5}};
    int B[M][N] = {0};
    int *d_A, *d_B;

    // Allocate device memory
    cudaMalloc((void **)&d_A, M * N * sizeof(int));
    cudaMalloc((void **)&d_B, M * N * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_A, A, M * N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(4, 4);
    dim3 blocksPerGrid(1, 1);
    complement_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, M, N);

    // Copy result back to host
    cudaMemcpy(B, d_B, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    printf("Output matrix B:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", B[i][j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}