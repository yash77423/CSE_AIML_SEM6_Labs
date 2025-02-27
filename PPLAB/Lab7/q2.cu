#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 1024 

__global__ void generateRS(char *d_s, char *d_rs, int len) {
    int idx = threadIdx.x;
    int start_pos = (len * idx) - (idx * (idx - 1)) / 2; 

    if (idx < len) {
        for (int j = 0; j < len - idx; j++) {
            d_rs[start_pos + j] = d_s[j];  
        }
    }
}

int main() {
    char *s = (char *)malloc(N * sizeof(char));  
    char *rs;  
    char *d_s, *d_rs;  

    // Input string
    printf("Enter string s: ");
    fgets(s, N, stdin);
    s[strcspn(s, "\n")] = 0; 

    int len = strlen(s);
    int outputSize = (len * (len + 1)) / 2; 

    rs = (char *)malloc(outputSize * sizeof(char)); 
    memset(rs, 0, outputSize * sizeof(char)); 

    cudaMalloc((void**)&d_s, len * sizeof(char));
    cudaMalloc((void**)&d_rs, outputSize * sizeof(char));

    cudaMemcpy(d_s, s, len * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemset(d_rs, 0, outputSize * sizeof(char)); 

    generateRS<<<1, len>>>(d_s, d_rs, len);

    cudaMemcpy(rs, d_rs, outputSize * sizeof(char), cudaMemcpyDeviceToHost);

    printf("Output string rs: %s\n", rs);

    // Free memory
    free(s);
    free(rs);
    cudaFree(d_s);
    cudaFree(d_rs);

    return 0;
}
