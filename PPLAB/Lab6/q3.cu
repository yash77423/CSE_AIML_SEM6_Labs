#include <stdio.h>
#include <cuda_runtime.h>

__global__ void odd_even_transposition_sort_kernel(int *arr, int n, int phase) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n - 1) return;

    // Odd phase (phase == 0) or Even phase (phase == 1)
    if ((idx % 2 == phase) && (idx < n - 1)) {
        if (arr[idx] > arr[idx + 1]) {
            // Swap elements
            int temp = arr[idx];
            arr[idx] = arr[idx + 1];
            arr[idx + 1] = temp;
        }
    }
}

int main() {
    int n = 1000;
    int *arr, *d_arr;

    arr = (int*)malloc(n * sizeof(int));

    // Initialize the array with random values
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;
    }

    // Allocate memory on device
    cudaMalloc((void**)&d_arr, n * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    // Number of iterations depends on the array size
    for (int phase = 0; phase < 2 * n; phase++) {
        // Kernel launch parameters
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;

        // Launch the kernel for each phase
        odd_even_transposition_sort_kernel<<<gridSize, blockSize>>>(d_arr, n, phase % 2);
    }

    // Copy the result back to host
    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the sorted array
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    // Free memory
    free(arr);
    cudaFree(d_arr);

    return 0;
}
