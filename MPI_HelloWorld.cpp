#include <stdio.h>
#include "mpi.h"                                 /* Include MPI header file that contains the library’s API */

int main (int argc, char* argv[])
{
  int my_rank, num_procs;

  MPI_Init (&argc, &argv);	                 /* start up MPI */
  MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);	 /* get current process' "ID" number (rank) */
  MPI_Comm_size (MPI_COMM_WORLD, &num_procs);	 /* get total number of processes */

  printf( "Hello from process %d of %d\n", my_rank, num_procs );

  MPI_Finalize();                                /* finish up MPI */

  return 0;
}
