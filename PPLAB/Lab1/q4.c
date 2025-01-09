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

    // Print the rank and toggled string
    char str[] = "HELLO";
    
    if (rank >= 0 && rank < size) {
        // Toggle the character at the position of the rank
        if (str[rank] >= 'a' && str[rank] <= 'z') {
            str[rank] -= 32;
        } else {
            str[rank] += 32;
        }
    }
    printf("Rank: %d Modified string: %s\n", rank, str);
    // printf("My rank is %d in total %d processes\n",rank,size);

    // Finalize the MPI environment
    MPI_Finalize();
    
    return 0;
}