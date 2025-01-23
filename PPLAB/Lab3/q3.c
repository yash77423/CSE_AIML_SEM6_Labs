#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <ctype.h>  // For isalpha() and tolower()

// Function to check if a character is a non-vowel
int is_non_vowel(char c) {
    char lower_c = tolower(c);
    return (isalpha(c) && !(lower_c == 'a' || lower_c == 'e' || lower_c == 'i' || lower_c == 'o' || lower_c == 'u'));
}

int main(int argc, char* argv[]) {
    int rank, size;
    char *input_string = NULL;
    int string_length, segment_length;
    int *local_count = NULL;
    int *global_counts = NULL;
    int total_non_vowels = 0;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Root process: Read the input string
    if (rank == 0) {
        printf("Enter a string: ");
        input_string = malloc(1000 * sizeof(char)); // Allocate space for the string (maximum 1000 characters)
        fgets(input_string, 1000, stdin);

        // Calculate string length (exclude the newline character at the end)
        string_length = strlen(input_string);
        if (input_string[string_length - 1] == '\n') {
            input_string[string_length - 1] = '\0';
            string_length--;
        }

        // Ensure the string length is divisible by the number of processes
        if (string_length % size != 0) {
            printf("Error: String length must be divisible by the number of processes.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast the string length to all processes
    MPI_Bcast(&string_length, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate the length of the segment for each process
    segment_length = string_length / size;

    // Allocate memory for each process to store its local count of non-vowels
    local_count = (int*) malloc(sizeof(int));

    // Scatter the string to all processes
    char *local_string = malloc(segment_length * sizeof(char));
    MPI_Scatter(input_string, segment_length, MPI_CHAR, local_string, segment_length, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Count the non-vowels in the local segment
    local_count[0] = 0;
    for (int i = 0; i < segment_length; i++) {
        if (is_non_vowel(local_string[i])) {
            local_count[0]++;
        }
    }

    // Gather the local counts from all processes
    global_counts = NULL;
    if (rank == 0) {
        global_counts = malloc(size * sizeof(int));
    }
    MPI_Gather(local_count, 1, MPI_INT, global_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Root process calculates the total count and prints results
    if (rank == 0) {
        printf("Non-vowels count by each process:\n");
        for (int i = 0; i < size; i++) {
            printf("Process %d: %d non-vowels\n", i, global_counts[i]);
            total_non_vowels += global_counts[i];
        }
        printf("Total non-vowels: %d\n", total_non_vowels);

        // Clean up the allocated memory
        free(input_string);
        free(global_counts);
    }

    // Clean up the allocated memory for local counts and strings
    free(local_count);
    free(local_string);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
