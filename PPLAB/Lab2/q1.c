#include "mpi.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]){
	int rank, size;
	char str[] = "Hello World";
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Status status;

	if(rank == 0){
		MPI_Ssend(str, strlen(str), MPI_CHAR,1,1,MPI_COMM_WORLD);
		fprintf(stdout,"I have sent \"%s\" from Process 0\n",str);
		fflush(stdout);
		MPI_Recv(str, strlen(str), MPI_CHAR,1,2,MPI_COMM_WORLD,&status);
		fprintf(stdout,"Received the Toggled String from Process 1: %s\n",str);
		fflush(stdout);
	}
	else{
		char buffer[12];
		buffer[11] = '\0';
		MPI_Recv(buffer, strlen(str), MPI_CHAR,0,1,MPI_COMM_WORLD,&status);
		for(int i = 0; i < strlen(buffer); i++){
			if(buffer[i] != ' '){
				if (buffer[i] >= 'a' && buffer[i] <= 'z') {
            		buffer[i] -= 32;
        		} else {
            		buffer[i] += 32;
        		}
			}
		}
		fprintf(stdout,"Toggled String from Process 1: %s\n",buffer);
		fflush(stdout);
		MPI_Ssend(buffer, strlen(buffer), MPI_CHAR,0,2,MPI_COMM_WORLD);
		fprintf(stdout,"Sent back the Toggled String from Process 1 to process 0: %s\n",buffer);
		fflush(stdout);
	}
	MPI_Finalize();
	return 0;
}

// to compile: mpicc q1.c -o q1.out
// to run: mpirun -np 2 ./q1.out

