#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MATRIX_SIZE 4  // 4x4 matrix

// Function to print a matrix
void print_matrix(int matrix[MATRIX_SIZE][MATRIX_SIZE]) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int matrix[MATRIX_SIZE][MATRIX_SIZE];  // Input matrix
    int output_matrix[MATRIX_SIZE][MATRIX_SIZE];  // Output matrix
    int local_input[MATRIX_SIZE];  // Each process will receive one row
    int local_output[MATRIX_SIZE];  // Each process will compute one row of output

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != MATRIX_SIZE) {
        if (rank == 0) {
            fprintf(stderr, "This program requires exactly %d processes!\n", MATRIX_SIZE);
        }
        MPI_Finalize();
        return 1;
    }

    // The root process reads the matrix and prints it
    if (rank == 0) {
        printf("Enter the 4x4 matrix:\n");
        for (int i = 0; i < MATRIX_SIZE; i++) {
            for (int j = 0; j < MATRIX_SIZE; j++) {
                scanf("%d", &matrix[i][j]);
            }
        }
        printf("Input matrix:\n");
        print_matrix(matrix);
    }

    // Scatter the rows of the input matrix to all processes
    MPI_Scatter(matrix, MATRIX_SIZE, MPI_INT, local_input, MATRIX_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform the scan operation to get the cumulative sum of rows
    // Start by copying the input row to the output row in each process
    MPI_Scan(local_input, local_output, MATRIX_SIZE, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Gather the rows of the output matrix back to the root process
    MPI_Gather(local_output, MATRIX_SIZE, MPI_INT, output_matrix, MATRIX_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    // The root process prints the output matrix
    if (rank == 0) {
        printf("Output matrix:\n");
        print_matrix(output_matrix);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
