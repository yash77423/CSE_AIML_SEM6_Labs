#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

__device__ int atomic_str_match(const char* sentence, const char* word, int sentence_len, int word_len) {
    int match = 1;
    for (int i = 0; i < word_len; i++) {
        if (sentence[i] != word[i]) {
            match = 0;
            break;
        }
    }
    return match;
}

__global__ void count_word_occurrences(const char* sentence, const char* word, int sentence_len, int word_len, int* count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx + word_len <= sentence_len) {
        int match = atomic_str_match(sentence + idx, word, sentence_len, word_len);
        if (match) {
            atomicAdd(count, 1);
        }
    }
}

int main() {
    const char* sentence = "hello world, hello CUDA world";
    const char* word = "hello";
    
    int sentence_len = strlen(sentence);
    int word_len = strlen(word);

    char *d_sentence, *d_word;
    int *d_count;
    int h_count = 0;

    // Allocate memory on device
    cudaMalloc((void**)&d_sentence, sentence_len + 1);
    cudaMalloc((void**)&d_word, word_len + 1);
    cudaMalloc((void**)&d_count, sizeof(int));

    // Copy data to device
    cudaMemcpy(d_sentence, sentence, sentence_len + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_word, word, word_len + 1, cudaMemcpyHostToDevice);
    cudaMemset(d_count, 0, sizeof(int));

    // Launch kernel with one block and enough threads
    int blockSize = 256;
    int numBlocks = (sentence_len + blockSize - 1) / blockSize;
    count_word_occurrences<<<numBlocks, blockSize>>>(d_sentence, d_word, sentence_len, word_len, d_count);

    // Copy the result back to host
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    printf("The word '%s' appears %d times in the sentence.\n", word, h_count);

    // Clean up
    cudaFree(d_sentence);
    cudaFree(d_word);
    cudaFree(d_count);

    return 0;
}
