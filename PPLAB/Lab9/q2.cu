#include <stdio.h>
#include <cuda.h>

#define M 3
#define N 4

__global__ void modify_matrix_kernel(float *A, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        float val = A[row * cols + col];
        for (int i = 0; i < row; i++) { // Raise to the power of (row + 1)
            val *= A[row * cols + col];
        }
        A[row * cols + col] = val;
    }
}

int main() {
    float A[M][N] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
    float *d_A;

    // Allocate device memory
    cudaMalloc((void **)&d_A, M * N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(N, M); // Each thread processes one element
    dim3 blocksPerGrid(1, 1);
    modify_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, M, N);

    // Copy result back to host
    cudaMemcpy(A, d_A, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    printf("Modified matrix:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", A[i][j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_A);

    return 0;
}