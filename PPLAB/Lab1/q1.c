#include "mpi.h"
#include <stdio.h>

int power(int a, int b){
    int power = 1;
    for(int i = 1; i <= b; i++){
        power*=a; 
    }
    return power;
}

int main(int argc, char *argv[]){
    int rank, size;

    // Initialize the MPI environment
    MPI_Init(&argc,&argv);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    // Get the total number of processes
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    // Print the rank and power of 2 for each process

    int x = 2;
    printf("Rank: %d, Power of 2: %d\n",rank,power(x,rank));

    // Finalize the MPI environment
    MPI_Finalize();
    
    return 0;
}