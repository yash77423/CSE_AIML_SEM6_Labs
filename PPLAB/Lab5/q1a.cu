#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(int *a, int *b, int *c, int N) {
    int index = threadIdx.x;
    if (index < N) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    int N = 1024; // Vector length
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;

    size_t size = N * sizeof(int);

    // Allocate memory for host arrays
    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);

    // Initialize vectors a and b with sample values
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i;
    }

    // Allocate memory for device arrays
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch kernel with block size N
    vectorAdd<<<1, N>>>(d_a, d_b, d_c, N);

    // Copy result from device to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < N; i++) {
        printf("%d ", c[i]);
    }
    printf("\n");

    // Free memory
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
