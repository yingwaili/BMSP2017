/*** An example to demonstrate a point-to-point communication using multiple ranks  ***/

#include <stdio.h>
#include "mpi.h"                                 /* Include MPI header file that contains the library’s API */

int main (int argc, char* argv[])
{
  int my_rank, num_procs;

  MPI_Init (&argc, &argv);	                 /* start up MPI */
  MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);	 /* get current process' "ID" number (rank) */
  MPI_Comm_size (MPI_COMM_WORLD, &num_procs);	 /* get total number of processes */
  MPI_Status status;                             /* stores the status of whether a communication is successful or not */

  int message;
  message = my_rank * 10;

  if (my_rank == 0) {     // Rank 0 receives messages from the others
    for (int i=1; i<num_procs; i++) {
      //MPI_Recv(&message, 1, MPI_INT, i, i, MPI_COMM_WORLD, &status);
      MPI_Recv(&message, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      printf("Received message %d from rank %d\n", message, status.MPI_SOURCE);
    }   
  }
  else                    // Every rank (except Rank 0) sends a message to Rank 0
    MPI_Send(&message, 1, MPI_INT, 0, my_rank, MPI_COMM_WORLD);

  MPI_Finalize();                                /* finish up MPI */

  return 0;
}
