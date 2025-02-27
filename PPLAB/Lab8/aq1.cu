#include <stdio.h>
#include <cuda_runtime.h>

#define M 2  // Number of rows
#define N 3  // Number of columns

// CUDA kernel to compute matrix B
__global__ void computeMatrixB(int *A, int *B, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int element = A[row * cols + col];

        // Compute row sum for even elements
        if (element % 2 == 0) {
            int rowSum = 0;
            for (int i = 0; i < cols; i++) {
                rowSum += A[row * cols + i];
            }
            B[row * cols + col] = rowSum;
        }
        // Compute column sum for odd elements
        else {
            int colSum = 0;
            for (int i = 0; i < rows; i++) {
                colSum += A[i * cols + col];
            }
            B[row * cols + col] = colSum;
        }
    }
}

int main() {
    int A[M][N] = {{1, 2, 3}, {4, 5, 6}};  // Input matrix
    int B[M][N];                            // Output matrix
    int *d_A, *d_B;

    // Allocate memory on device
    cudaMalloc((void**)&d_A, M * N * sizeof(int));
    cudaMalloc((void**)&d_B, M * N * sizeof(int));

    // Copy input matrix A to device
    cudaMemcpy(d_A, A, M * N * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 threadsPerBlock(N, M);
    dim3 blocksPerGrid(1, 1);

    // Launch kernel
    computeMatrixB<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, M, N);

    // Copy result back to host
    cudaMemcpy(B, d_B, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print input matrix
    printf("Input Matrix A:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", A[i][j]);
        }
        printf("\n");
    }

    // Print output matrix
    printf("Output Matrix B:\n");
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