#include <stdio.h>
#include <cuda_runtime.h>

#define WIDTH 8
#define MASK_WIDTH 3
#define TILE_WIDTH 4

__global__ void tiledConvolution1D(int *N, int *M, int *P, int width) {
    __shared__ int N_s[TILE_WIDTH + MASK_WIDTH - 1];
    int tid = threadIdx.x;
    int offset = blockIdx.x * TILE_WIDTH;

    if (offset + tid < width) {
        N_s[tid] = N[offset + tid];
    } else {
        N_s[tid] = 0;
    }
    __syncthreads();

    if (offset + tid < width) {
        int sum = 0;
        for (int j = 0; j < MASK_WIDTH; j++) {
            if (tid + j < TILE_WIDTH + MASK_WIDTH - 1) {
                sum += N_s[tid + j] * M[j];
            }
        }
        P[offset + tid] = sum;
    }
}

int main() {
    int *N, *M, *P;
    int *d_N, *d_M, *d_P;
    int size_N = WIDTH * sizeof(int);
    int size_M = MASK_WIDTH * sizeof(int);

    N = (int*)malloc(size_N);
    M = (int*)malloc(size_M);
    P = (int*)malloc(size_N);

    printf("Enter input array (size %d):\n", WIDTH);
    for (int i = 0; i < WIDTH; i++) scanf("%d", &N[i]);

    printf("Enter mask (size %d):\n", MASK_WIDTH);
    for (int i = 0; i < MASK_WIDTH; i++) scanf("%d", &M[i]);

    cudaMalloc(&d_N, size_N);
    cudaMalloc(&d_M, size_M);
    cudaMalloc(&d_P, size_N);

    cudaMemcpy(d_N, N, size_N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, M, size_M, cudaMemcpyHostToDevice);

    tiledConvolution1D<<<(WIDTH + TILE_WIDTH - 1) / TILE_WIDTH, TILE_WIDTH>>>(d_N, d_M, d_P, WIDTH);

    cudaMemcpy(P, d_P, size_N, cudaMemcpyDeviceToHost);

    printf("Convolution Result:\n");
    for (int i = 0; i < WIDTH; i++) printf("%d ", P[i]);
    printf("\n");

    cudaFree(d_N); cudaFree(d_M); cudaFree(d_P);
    free(N); free(M); free(P);
    return 0;
}