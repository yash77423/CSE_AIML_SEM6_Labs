#include <stdio.h>
#include <string.h>
#include <cuda.h>

// Kernel to reverse each word in the string
__global__ void reverseWords(char *str, int *wordIndices, int numWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWords) {
        int start = wordIndices[idx];
        int end = wordIndices[idx + 1] - 1;
        while (start < end) {
            char temp = str[start];
            str[start] = str[end];
            str[end] = temp;
            start++;
            end--;
        }
    }
}

int main() {
    char h_str[] = "Hello World CUDA Programming";
    int len = strlen(h_str);
    int numWords = 0;

    // Display the original string
    printf("Original String: %s\n", h_str);

    // Count the number of words and store their start indices
    int wordIndices[100]; // Assuming max 100 words
    wordIndices[0] = 0;
    for (int i = 0; i < len; i++) {
        if (h_str[i] == ' ') {
            numWords++;
            wordIndices[numWords] = i + 1;
        }
    }
    numWords++;
    wordIndices[numWords] = len;

    // Allocate device memory
    char *d_str;
    int *d_wordIndices;
    cudaMalloc((void **)&d_str, len + 1);
    cudaMalloc((void **)&d_wordIndices, (numWords + 1) * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_str, h_str, len + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_wordIndices, wordIndices, (numWords + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numWords + threadsPerBlock - 1) / threadsPerBlock;
    reverseWords<<<blocksPerGrid, threadsPerBlock>>>(d_str, d_wordIndices, numWords);

    // Copy result back to host
    cudaMemcpy(h_str, d_str, len + 1, cudaMemcpyDeviceToHost);

    // Print the result
    printf("Reversed String: %s\n", h_str);

    // Free device memory
    cudaFree(d_str);
    cudaFree(d_wordIndices);

    return 0;
}