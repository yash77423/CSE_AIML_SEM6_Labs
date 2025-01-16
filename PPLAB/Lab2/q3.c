#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;

    int arr[size];
    int num;
    int buffer_size = size * (MPI_BSEND_OVERHEAD + sizeof(int));  
    void* buffer = malloc(buffer_size);
    
    MPI_Buffer_attach(buffer, buffer_size);

    if (rank == 0) {
        fprintf(stdout, "Enter the elements for the array:\n");
        for (int i = 1; i < size; i++) {
            scanf("%d", &arr[i]);
        }
        for (int i = 1; i < size; i++) {
            num = arr[i];
            MPI_Bsend(&num, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            fprintf(stdout, "Sent number %d from Process %d\n", arr[i], rank);
            fflush(stdout);
        }
    }
    else if (rank % 2 == 0) {
        int x;
        MPI_Recv(&x, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        fprintf(stdout, "Process: %d Square: %d\n", rank, x * x);
        fflush(stdout);
    }
    else {
        int y;
        MPI_Recv(&y, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        fprintf(stdout, "Process: %d Cube: %d\n", rank, y, y * y * y);
        fflush(stdout);
    }

    MPI_Buffer_detach(&buffer, &buffer_size);
    // free(buffer);

    MPI_Finalize();

    return 0;
}
