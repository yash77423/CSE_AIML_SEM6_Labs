#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MATRIX_SIZE 3

int main(int argc, char *argv[]) {
    int rank, size;
    int matrix[MATRIX_SIZE][MATRIX_SIZE];
    int search_element, local_count = 0, total_count = 0;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Root process: Read the matrix
        printf("Enter a 3x3 matrix (9 elements):\n");
        for (int i = 0; i < MATRIX_SIZE; i++) {
            for (int j = 0; j < MATRIX_SIZE; j++) {
                scanf("%d", &matrix[i][j]);
            }
        }

        // Root process: Enter the element to be searched
        printf("Enter the element to search for: ");
        scanf("%d", &search_element);
    }

    // Broadcast the matrix and search element to all processes
    MPI_Bcast(&search_element, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(matrix, MATRIX_SIZE * MATRIX_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    // Divide the work among the processes
    int elements_per_process = MATRIX_SIZE * MATRIX_SIZE / 3;
    int start_index = rank * elements_per_process;
    int end_index = (rank + 1) * elements_per_process;

    if (rank == 2) {
        // The last process handles any leftover elements
        end_index = MATRIX_SIZE * MATRIX_SIZE;
    }

    // Count occurrences of the search_element in the assigned portion
    for (int i = start_index; i < end_index; i++) {
        int row = i / MATRIX_SIZE;
        int col = i % MATRIX_SIZE;
        if (matrix[row][col] == search_element) {
            local_count++;
        }
    }

    // Use MPI_Reduce to sum up the local counts at the root process
    MPI_Reduce(&local_count, &total_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Root process prints the result
    if (rank == 0) {
        printf("The element %d occurred %d times in the matrix.\n", search_element, total_count);
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
