#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {

    int rank, size;
    int arr[size];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;

    int buffer_size = size * sizeof(int);  
    int* buffer = (int*)malloc(buffer_size);
    
    MPI_Buffer_attach(buffer, buffer_size);

    if (rank == 0) {
        fprintf(stdout, "Enter the elements for the array:\n");
        for (int i = 1; i < size; i++) {
            scanf("%d", &arr[i]);
        }
        for (int i = 1; i < size; i++) {
            MPI_Bsend(&arr[i], 1, MPI_INT, i, 1, MPI_COMM_WORLD);  
            MPI_Buffer_detach(&buffer, &buffer_size);
            free(buffer);

            fprintf(stdout, "My rank is %d and I sent %d to process %d\n", rank, arr[i], i);
            fflush(stdout);
        }
    }
    else if (rank % 2 == 0) {
        int x;
        MPI_Recv(&x, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        fprintf(stdout, "My rank is %d and I received %d, squared value is : %d\n", rank, x, x * x);
        fflush(stdout);
    }
    else {
        int y;
        MPI_Recv(&y, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        fprintf(stdout, "My rank is %d and I received %d, cubed value is : %d\n", rank, y, y * y * y);
        fflush(stdout);
    }

    // MPI_Buffer_detach(&buffer, &buffer_size);
    // free(buffer);

    MPI_Finalize();

    return 0;
}
