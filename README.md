# This is a set of demonstration codes complementary to the Replica-Exchange Wang-Landau sampling tutorial lectures given at the IX Brazilian Meeting on Simulational Physics, August 21-25, 2017, Natal, Brazil. 

# To compile the code:
mpicxx -o WLpotts_mpi WLpotts_mpi.cpp 

# Some examples to run the code:
Example 1:
3 MPI ranks, 0.5 overlap, 1 walker/window, 1000 MC steps between replica exchanges, random number seed = 14
mpirun -np 3 ./WLpotts_mpi 0.5 1 1000 14

Example 2:
6 MPI ranks, 0.5 overlap, 2 walkers/window, 1000 MC steps between replica exchanges, random number seed = 14
mpirun -np 6 ./WLpotts_mpi 0.5 2 1000 14

Example 3:
5 MPI ranks, 0.8 overlap, 1 walkers/window, 500 MC steps between replica exchanges, random number seed = 398
mpirun -np 5 ./WLpotts_mpi 0.8 1 500 398
