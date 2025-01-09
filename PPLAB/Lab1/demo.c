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

    // Print the rank and size of each process
    printf("My rank is %d in total %d processes\n",rank,size);

    // Finalize the MPI environment
    MPI_Finalize();
    
    return 0;
}