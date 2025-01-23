#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Function to calculate the factorial of a number
int factorial(int n) {
    int result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}

int main(int argc, char* argv[]) {
    int rank, size;
    int N;
    int *values = NULL;
    int *results = NULL;
    int sum = 0;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Root process: read N values
    if (rank == 0) {
        printf("Enter the number of values (N): ");
        scanf("%d", &N);
        values = (int*) malloc(N * sizeof(int));
        
        // Read the N values
        printf("Enter %d values:\n", N);
        for (int i = 0; i < N; i++) {
            scanf("%d", &values[i]);
        }
    }

    // Broadcast the number of values to all processes
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate space for the result on each process
    int value = 0;

    MPI_Scatter(values, 1, MPI_INT, &value, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute factorial of the received value
    int fact = factorial(value);

    // Gather the results on the root process
    if (rank == 0) {
        results = (int*) malloc(N * sizeof(int));
        results[0] = fact; // Root process stores its own result
        MPI_Gather(&fact, 1, MPI_INT, results, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gather(&fact, 1, MPI_INT, results, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Root process computes the sum of all factorials
    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            sum += results[i];
        }
        printf("Sum of all factorials: %d\n", sum);
    }

    // Clean up memory
    if (rank == 0) {
        free(values);
        free(results);
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
