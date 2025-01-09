#include "mpi.h"
#include <stdio.h>

int fact(int n){
    if(n == 0) return 1;
    return n*fact(n-1);
}

int fib(int n){
    if (n == 0) return 0;
    if (n == 1) return 1;
    return fib(n-1) + fib(n-2);
}

int main(int argc, char *argv[]){
    int rank, size;

    // Initialize the MPI environment
    MPI_Init(&argc,&argv);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    // Get the total number of processes
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    // Print the rank and factorial or fibonacci based on each process
    printf("my Rank: %d\n", rank);
    if(rank % 2 == 0){
        printf(" Rank: %d Factorial: %d\n", rank,fact(rank));
    }
    else{
        printf("Rank: %d kth Fibonacci: %d\n", rank, fib(rank));
    }

    // Finalize the MPI environment
    MPI_Finalize();
    
    return 0;
}