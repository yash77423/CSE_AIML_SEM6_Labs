#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

// Function to merge S1 and S2 alternately for a given portion
void merge_strings(char *s1, char *s2, char *result, int segment_length) {
    for (int i = 0; i < segment_length; i++) {
        result[2*i] = s1[i];     // Even index: take from S1
        result[2*i+1] = s2[i];   // Odd index: take from S2
    }
}

int main(int argc, char* argv[]) {
    int rank, size;
    char *S1 = NULL, *S2 = NULL;
    int string_length, segment_length;
    char *local_S1 = NULL, *local_S2 = NULL;
    char *local_result = NULL;
    char *final_result = NULL;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Root process: Read the input strings
    if (rank == 0) {
        printf("Enter string S1: ");
        S1 = malloc(1000 * sizeof(char)); // Allocate space for the string (max 1000 characters)
        fgets(S1, 1000, stdin);
        S1[strcspn(S1, "\n")] = '\0'; // Remove newline character

        printf("Enter string S2: ");
        S2 = malloc(1000 * sizeof(char)); // Allocate space for the string (max 1000 characters)
        fgets(S2, 1000, stdin);
        S2[strcspn(S2, "\n")] = '\0'; // Remove newline character

        // Get the length of the strings
        string_length = strlen(S1);
        
        // Ensure both strings are of the same length
        if (string_length != strlen(S2)) {
            printf("Error: Strings must have the same length.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Ensure the string length is divisible by the number of processes
        if (string_length % size != 0) {
            printf("Error: String length must be divisible by the number of processes.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast the string length to all processes
    MPI_Bcast(&string_length, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate the length of the segment each process will handle
    segment_length = string_length / size;

    // Allocate memory for each process to store its part of S1, S2, and result
    local_S1 = malloc(segment_length * sizeof(char));
    local_S2 = malloc(segment_length * sizeof(char));
    local_result = malloc(2 * segment_length * sizeof(char)); // Result will be twice the size of the segment

    // Scatter the string segments to all processes
    MPI_Scatter(S1, segment_length, MPI_CHAR, local_S1, segment_length, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(S2, segment_length, MPI_CHAR, local_S2, segment_length, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Merge the local segments from S1 and S2
    merge_strings(local_S1, local_S2, local_result, segment_length);

    // Gather the results from all processes
    if (rank == 0) {
        final_result = malloc(string_length * sizeof(char));
    }
    MPI_Gather(local_result, 2 * segment_length, MPI_CHAR, final_result, 2 * segment_length, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Root process prints the final result
    if (rank == 0) {
        printf("Resultant String: %s\n", final_result);
        // Clean up the allocated memory
        free(S1);
        free(S2);
        free(final_result);
    }

    // Clean up the allocated memory for local segments and results
    free(local_S1);
    free(local_S2);
    free(local_result);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
