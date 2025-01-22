#include <stdio.h>
#include "mpi.h"

int main(int argc, char* argv[]){

    int rank, size, x;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;

    if(rank == 0){
        printf("Enter an integer value: ");
        scanf("%d", &x);
        MPI_Send(&x, 1, MPI_INT, 1, 1, MPI_COMM_WORLD);
        MPI_Recv(&x, 1, MPI_INT, size-1, 1, MPI_COMM_WORLD, &status);
        printf("Final value received: %d\n", x);
    }
    else{
        MPI_Recv(&x, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD, &status);
        printf("Value received by Process %d: %d\n", rank, x);
        x = x + 1;
        printf("New value of x: %d\n", x);

        if(rank < size - 1){
            MPI_Send(&x, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD);
        }
        else if(rank == size-1){
            MPI_Send(&x, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();

    return 0;
}