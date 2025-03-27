#include <stdio.h>
#include <cuda_runtime.h>

#define N 8
#define MASK_WIDTH 3

__constant__ int mask[MASK_WIDTH];

__global__ void convolution1D(int *input, int *output, int width) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < width) {
        int sum = 0;
        for (int i = 0; i < MASK_WIDTH; i++) {
            int offset = tid - MASK_WIDTH / 2 + i;
            if (offset >= 0 && offset < width) {
                sum += input[offset] * mask[i];
            }
        }
        output[tid] = sum;
    }
}

int main() {
    int *input, *output;
    int *d_input, *d_output;
    int h_mask[MASK_WIDTH] = {1, 2, 1}; // Example mask (Gaussian blur)
    int size = N * sizeof(int);

    input = (int*)malloc(size);
    output = (int*)malloc(size);

    printf("Enter input array (size %d):\n", N);
    for (int i = 0; i < N; i++) scanf("%d", &input[i]);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask, h_mask, MASK_WIDTH * sizeof(int));

    convolution1D<<<(N + 255) / 256, 256>>>(d_input, d_output, N);

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    printf("Convolution Result:\n");
    for (int i = 0; i < N; i++) printf("%d ", output[i]);
    printf("\n");

    cudaFree(d_input); cudaFree(d_output);
    free(input); free(output);
    return 0;
}