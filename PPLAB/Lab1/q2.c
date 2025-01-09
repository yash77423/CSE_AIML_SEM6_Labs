#include "mpi.h"
#include <stdio.h>

int main(int argc, char *argv[]){
    int rank, size;

    // Initialize the MPI environment
    MPI_Init(&argc,&argv);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    // Get the total number of processes
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    // Print the rank and "Hello"/"World" based on each process
    if(rank%2 == 0){
        printf("Rank: %d Hello\n",rank);
    }
    else{
        printf("Rank: %d World\n",rank);
    }

    // Finalize the MPI environment
    MPI_Finalize();
    
    return 0;
}