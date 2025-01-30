#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Function to compute factorial using int
int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

// Error handling function
void check_mpi_error(int mpi_err_code, const char *err_msg) {
    if (mpi_err_code != MPI_SUCCESS) {
        char error_string[100];
        int error_class;
        MPI_Error_class(mpi_err_code, &error_class);
        MPI_Error_string(mpi_err_code, error_string, NULL);
        fprintf(stderr, "MPI error %d: %s - %s\n", error_class, error_string, err_msg);
        MPI_Abort(MPI_COMM_WORLD, mpi_err_code);
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int N;
    int local_fact, global_sum;

    // Initialize MPI
    int mpi_err_code = MPI_Init(&argc, &argv);
    check_mpi_error(mpi_err_code, "MPI_Init failed");

    // Get the rank and size of the process
    mpi_err_code = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    check_mpi_error(mpi_err_code, "MPI_Comm_rank failed");

    mpi_err_code = MPI_Comm_size(MPI_COMM_WORLD, &size);
    check_mpi_error(mpi_err_code, "MPI_Comm_size failed");

    // Read the value of N (number of terms in the sum) from the root process
    if (rank == 0) {
        printf("Enter the value of N: ");
        fflush(stdout);
        scanf("%d", &N);
    }

    // Broadcast N to all processes
    mpi_err_code = MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    check_mpi_error(mpi_err_code, "MPI_Bcast failed");

    // Calculate factorial for the current rank
    if (rank < N) {
        local_fact = factorial(rank + 1);  // rank starts from 0, but factorial is 1-indexed
    } else {
        local_fact = 0;  // If rank is greater than N-1, no factorial to contribute
    }

    // Use MPI_Scan to compute partial sums of factorials
    mpi_err_code = MPI_Scan(&local_fact, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    check_mpi_error(mpi_err_code, "MPI_Scan failed");

    // The last process will have the final result
    if (rank == size - 1) {
        printf("The sum of factorials from 1! to %d! is: %d\n", N, global_sum);
    }

    // Finalize MPI
    mpi_err_code = MPI_Finalize();
    check_mpi_error(mpi_err_code, "MPI_Finalize failed");

    return 0;
}
