#include <stdio.h>
#include <cuda.h>

#define N 4 // Size of the matrix and vector

__global__ void spmv_csr_kernel(int *row_ptr, int *col_idx, float *values, float *x, float *y, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float dot = 0;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        for (int i = row_start; i < row_end; i++) {
            dot += values[i] * x[col_idx[i]];
        }
        y[row] = dot;
    }
}

int main() {
    // Input sparse matrix in CSR format
    int row_ptr[] = {0, 2, 4, 7, 8};
    int col_idx[] = {0, 1, 1, 2, 0, 2, 3, 1};
    float values[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float x[N] = {1, 2, 3, 4}; // Input vector
    float y[N] = {0}; // Output vector

    // Allocate device memory
    int *d_row_ptr, *d_col_idx;
    float *d_values, *d_x, *d_y;
    cudaMalloc((void **)&d_row_ptr, (N + 1) * sizeof(int));
    cudaMalloc((void **)&d_col_idx, 8 * sizeof(int));
    cudaMalloc((void **)&d_values, 8 * sizeof(float));
    cudaMalloc((void **)&d_x, N * sizeof(float));
    cudaMalloc((void **)&d_y, N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_row_ptr, row_ptr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, col_idx, 8 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, 8 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    spmv_csr_kernel<<<1, N>>>(d_row_ptr, d_col_idx, d_values, d_x, d_y, N);

    // Copy result back to host
    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    printf("Result vector y:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", y[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}