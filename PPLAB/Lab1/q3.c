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

    // Print the rank and do calculator operations for each process

    int a = 2, b = 3;
    switch(rank){
    case 0:
        printf("Rank: %d  ", rank);
        printf("  %d + %d = %d\n", a, b, a+b);
        break;
    case 1:
        printf("Rank: %d  ", rank);
        printf("  %d - %d = %d\n", a, b, a-b);
        break;
    case 2:
        printf("Rank: %d  ", rank);
        printf("  %d * %d = %d\n", a, b, a*b);
        break;
    case 3:
        printf("Rank: %d  ", rank);
        printf("  %d / %d = %d\n", a, b, a/b);
        break;
    }

    // Finalize the MPI environment
    MPI_Finalize();
    
    return 0;
}