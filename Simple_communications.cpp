/*** An example to demonstrate a point-to-point communication  ***/

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

  if (my_rank == 0)        // Rank 0 receives a message from Rank 1
  {
    MPI_Recv(&message, 1, MPI_INT, 1, 2017, MPI_COMM_WORLD, &status);
    printf("I am Rank 0. Received message %d from Rank %d\n", message, status.MPI_SOURCE);
  }
  else if (my_rank == 1)   // Rank 1 sends a message to Rank 0
  {
    MPI_Send(&message, 1, MPI_INT, 0, 2017, MPI_COMM_WORLD);
    printf("I am Rank 1. Sent a message to Rank 0.\n");
  }
  else                       // Other ranks sit and wait
    printf("I am Rank %d. I have nothing to do.\n", my_rank);

  MPI_Finalize();                                /* finish up MPI */

  return 0;
}
