#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size, M, N;
    int *arr = NULL;
    int *recv_buffer = NULL;
    float local_avg, total_avg;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Root process (rank 0)
    if (rank == 0) {
        printf("Enter the number of processes (N): ");
        scanf("%d", &N);

        printf("Enter the number of elements per process (M): ");
        scanf("%d", &M);

        // Check if N matches the number of processes
        if (size != N) {
            printf("Error: The number of processes must match N.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Allocate array to hold N * M elements
        arr = (int*) malloc(N * M * sizeof(int));

        // Read N * M elements into the array
        printf("Enter the %d elements:\n", N * M);
        for (int i = 0; i < N * M; i++) {
            scanf("%d", &arr[i]);
        }
    }

    // Broadcast M to all processes
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate buffer for each process to receive M elements
    recv_buffer = (int*) malloc(M * sizeof(int));

    // Scatter the N * M array to all processes
    MPI_Scatter(arr, M, MPI_INT, recv_buffer, M, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate the local average for the M elements each process receives
    int sum = 0;
    for (int i = 0; i < M; i++) {
        sum += recv_buffer[i];
    }
    local_avg = (float)sum / M;

    // Gather all local averages in the root process
    float *avg_buffer = NULL;
    if (rank == 0) {
        avg_buffer = (float*) malloc(N * sizeof(float));
    }
    MPI_Gather(&local_avg, 1, MPI_FLOAT, avg_buffer, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Root process calculates the total average
    if (rank == 0) {
        total_avg = 0.0f;
        for (int i = 0; i < N; i++) {
            total_avg += avg_buffer[i];
        }
        total_avg /= N;

        // Print the total average
        printf("Total average: %.2f\n", total_avg);

        // Clean up the allocated memory
        free(arr);
        free(avg_buffer);
    }

    // Clean up the allocated memory for recv_buffer
    free(recv_buffer);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
