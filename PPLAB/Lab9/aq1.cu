#include <stdio.h>
#include <cuda.h>

#define M 2
#define N 3

__global__ void row_col_sum_kernel(int *A, int *B, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        int sum = 0;
        for (int i = 0; i < cols; i++) sum += A[row * cols + i]; // Row sum
        for (int i = 0; i < rows; i++) sum += A[i * cols + col]; // Column sum
        B[row * cols + col] = sum;
    }
}

int main() {
    int A[M][N] = {{1, 2, 3}, {4, 5, 6}};
    int B[M][N] = {0};
    int *d_A, *d_B;

    // Allocate device memory
    cudaMalloc((void **)&d_A, M * N * sizeof(int));
    cudaMalloc((void **)&d_B, M * N * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_A, A, M * N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(3, 2);
    dim3 blocksPerGrid(1, 1);
    row_col_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, M, N);

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