#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ void concatenateString(char *Sin, char *Sout, int N, int sinLength) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalLength = sinLength * N;
    
    if (idx < totalLength) {
        int modIdx = idx % sinLength;
        Sout[idx] = Sin[modIdx];
    }
}

void concatenate(char *Sin, int N) {
    int sinLength = strlen(Sin);
    int totalLength = sinLength * N;
    char *d_Sin, *d_Sout;
    
    char *Sout = (char*)malloc(totalLength + 1);  // Extra space for null terminator
    cudaMalloc((void**)&d_Sin, sinLength * sizeof(char));
    cudaMalloc((void**)&d_Sout, totalLength * sizeof(char));
    
    cudaMemcpy(d_Sin, Sin, sinLength * sizeof(char), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalLength + threadsPerBlock - 1) / threadsPerBlock;
    
    concatenateString<<<blocksPerGrid, threadsPerBlock>>>(d_Sin, d_Sout, N, sinLength);
    
    cudaMemcpy(Sout, d_Sout, totalLength * sizeof(char), cudaMemcpyDeviceToHost);
    Sout[totalLength] = '\0';  // Null terminate the string
    
    printf("Output String: %s\n", Sout);
    
    cudaFree(d_Sin);
    cudaFree(d_Sout);
    free(Sout);
}

int main() {
    char Sin[] = "Hello";
    int N = 3;
    
    printf("Input String: %s\n", Sin);
    concatenate(Sin, N);
    
    return 0;
}
