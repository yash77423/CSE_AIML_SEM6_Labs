#include <stdio.h>
#include <cuda_runtime.h>

#define N 4  // Size of the matrix (N x N)

// Function to compute factorial
__device__ int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

// Function to compute sum of digits
__device__ int sumOfDigits(int n) {
    int sum = 0;
    while (n != 0) {
        sum += n % 10;
        n /= 10;
    }
    return sum;
}

// CUDA kernel to transform the matrix
__global__ void transformMatrix(int *A, int *B, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        if (row == col) {
            // Principal diagonal: replace with 0
            B[row * size + col] = 0;
        } else if (row < col) {
            // Above diagonal: replace with factorial
            B[row * size + col] = factorial(A[row * size + col]);
        } else {
            // Below diagonal: replace with sum of digits
            B[row * size + col] = sumOfDigits(A[row * size + col]);
        }
    }
}

int main() {
    int A[N][N] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    };  // Input matrix
    int B[N][N];  // Output matrix
    int *d_A, *d_B;

    // Allocate memory on device
    cudaMalloc((void**)&d_A, N * N * sizeof(int));
    cudaMalloc((void**)&d_B, N * N * sizeof(int));

    // Copy input matrix A to device
    cudaMemcpy(d_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid(1, 1);

    // Launch kernel
    transformMatrix<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);

    // Copy result back to host
    cudaMemcpy(B, d_B, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print input matrix
    printf("Input Matrix A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d\t", A[i][j]);
        }
        printf("\n");
    }

    // Print output matrix
    printf("Output Matrix B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d\t", B[i][j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}