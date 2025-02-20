#include <stdio.h>
#include <cuda_runtime.h>

__global__ void selection_sort_kernel(int *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int minIdx = idx;
    // Find the minimum element in the unsorted portion
    for (int j = idx + 1; j < n; j++) {
        if (arr[j] < arr[minIdx]) {
            minIdx = j;
        }
    }

    // Swap the current element with the minimum element
    if (minIdx != idx) {
        int temp = arr[idx];
        arr[idx] = arr[minIdx];
        arr[minIdx] = temp;
    }
}

int main() {
    int n = 1000; // Size of the array
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

    // Kernel launch parameters
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch the kernel to perform selection sort
    selection_sort_kernel<<<gridSize, blockSize>>>(d_arr, n);

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
