#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define WIDTH 4

__global__ void matrixMul(int *A, int *B, int *C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        int sum = 0;
        for (int k = 0; k < width; k++) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

int main() {
    int *A, *B, *C;
    int *d_A, *d_B, *d_C;
    int size = WIDTH * WIDTH * sizeof(int);

    A = (int*)malloc(size);
    B = (int*)malloc(size);
    C = (int*)malloc(size);

    printf("Enter Matrix A (4x4):\n");
    for (int i = 0; i < WIDTH * WIDTH; i++) scanf("%d", &A[i]);

    printf("Enter Matrix B (4x4):\n");
    for (int i = 0; i < WIDTH * WIDTH; i++) scanf("%d", &B[i]);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE, (WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrixMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, WIDTH);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Result Matrix C:\n");
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < WIDTH; j++) {
            printf("%d ", C[i * WIDTH + j]);
        }
        printf("\n");
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A); free(B); free(C);
    return 0;
}