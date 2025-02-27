#include <iostream>
#include <cuda_runtime.h>

#define N 4  // Matrix size (N x N)

__global__ void matrixMulRow(int *A, int *B, int *C, int width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < width) {
        for (int col = 0; col < width; col++) {
            C[row * width + col] = 0;
            for (int k = 0; k < width; k++) {
                C[row * width + col] += A[row * width + k] * B[k * width + col];
            }
        }
    }
}

__global__ void matrixMulCol(int *A, int *B, int *C, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width) {
        for (int row = 0; row < width; row++) {
            C[row * width + col] = 0;
            for (int k = 0; k < width; k++) {
                C[row * width + col] += A[row * width + k] * B[k * width + col];
            }
        }
    }
}

__global__ void matrixMulElement(int *A, int *B, int *C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < width && col < width) {
        C[row * width + col] = 0;
        for (int k = 0; k < width; k++) {
            C[row * width + col] += A[row * width + k] * B[k * width + col];
        }
    }
}

int main() {
    int A[N][N], B[N][N], C[N][N];
    int *d_A, *d_B, *d_C;

    // Initialize matrices A and B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = i + j;
            B[i][j] = i - j;
        }
    }

    // Allocate memory on device
    cudaMalloc((void**)&d_A, N * N * sizeof(int));
    cudaMalloc((void**)&d_B, N * N * sizeof(int));
    cudaMalloc((void**)&d_C, N * N * sizeof(int));

    // Copy matrices A and B to device
    cudaMemcpy(d_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel (2a)
    matrixMulRow<<<1, N>>>(d_A, d_B, d_C, N);

    // Kernel launch (2b)
    // matrixMulCol<<<1, N>>>(d_A, d_B, d_C, N);

    // Kernel launch (2c)
    // dim3 threadsPerBlock(N, N);
    // matrixMulElement<<<1, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(C, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}