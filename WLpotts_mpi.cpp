/*
  Replica Exchange Wang Landau demo code for simulating the 2D Potts model
  (c) Thomas Vogel and Ying Wai Li (2013, 2014, 2015, 2016, 2017)
 
  License: Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
           https://creativecommons.org/licenses/by-sa/4.0/legalcode
	   
	   You are free to:

	   - Share, copy and redistribute the material in any medium or format
	   - Adapt, transform, and build upon the material for any purpose, even commercially

	   (The licensor cannot revoke these freedoms as long as you
	   follow the following license terms.)

	   Under the following terms:

	   1) Attribution: You must give appropriate credit and must
	   not remove this preface from any file containing parts of
	   this code. You may give credit in any reasonable manner,
	   but not in any way that suggests the licensor endorses you
	   or your use.

	   If you publish data that was created using this code, or
	   parts of it, we ask to cite the original papers:
	     - T. Vogel et al., Phys. Rev. Lett. 110 (2013) 210603
             - T. Vogel et al., Phys. Rev. E 90 (2014) 023302
	   Accompanying publications include:
	     - T. Vogel et al., J. Phys.: Conf. Ser. 487 (2014) 012001
             - Y.W. Li et al., J. Phys.: Conf. Ser. 510 (2014) 012012

	   2) ShareAlike: If you modify, transform, or build upon the
	   material, you must distribute your contributions under the
	   same license as the original.

	   3) No additional restrictions: You may not apply legal terms
	   or technological measures that legally restrict others from
	   doing anything the license permits.

	   Notices:

	   You do not have to comply with the license for elements of
	   the material in the public domain or where your use is
	   permitted by an applicable exception or limitation.  No
	   warranties are given. The license may not give you all of
	   the permissions necessary for your intended use. For
	   example, other rights such as publicity, privacy, or moral
	   rights may limit how you use the material.

 */

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <limits.h>

const int L1dim = 4;                            // linear dimension 
const int Dimension = 2;                        // not sure at this moment if other than 2D is actually implemented 
const int numberspins = pow(L1dim, Dimension);  // total number of spins
const int numberneighbors = 4;                  // for 2D square lattice (others not implemented right now)
const int Eglobalmin = -2*numberspins;          // minimum energy for 2D square lattice Potts model
const int Eglobalmax = 0;                       // maximum energy of Potts model
const int bctype = 0;                           // type of boundary condition: 0 - periodic; 1 - Braskamp Kunz
const int q = 10;                               // number of different possible spin states (q-state Potts model)

int *latticepoint;                      // list containing values of all spins
int *neighbor;                          // list containing indices of neighbors for all spins

double *HE;                             // energy histogram
double *lngE;                           // ln g(E) estimator
double *lngE_buf;                       // ln g(E) estimator buffer for exchange
int hist_size = 2*numberspins+1;        // histogram size


int rseed;                              // seed for random number generator
int energy, localenergy;

double Emin,Emax;                       // doubles as calculation of boundaries in Energy is not in 'int'
int Eminindex, Emaxindex, Estartindex;  // local boundaries as index 

// MPI; set up with local communicators for replica exchange (RE)
// Needed so that two processes can talk w/o depending on / bothering the others
int numprocs, myid, multiple, comm_id;
// each process belongs to two local groups,
// one to communicate to left neighbor and one to communicate to right neighbor
// each process has different loca IDs in different communicatore, in general 
int mylocalid[2];                       // id in local communicators 
MPI_Comm *mpi_local_comm;
MPI_Group *mpi_local_group;
MPI_Group world;
MPI_Status status;
int merge_hists = 1;                    // flag whether or not to merge histograms between iterations for multiple walkers on the same energy range

// to keep track of exchange statistics
int tryleft, tryright, exchangeleft, exchangeright;

// File handlers for I/O
FILE *file;
FILE *stdoutlog;
FILE *wanderlog;
char filename[50];
char stdoutlogname[128];
char wanderlogname[128];
int ret_status;

// to track execution times
time_t timenow, timestart, timeend;


void keypressed()     // just for developing / manual debugging
{
  for (;;)
    if (getchar() == 27) break;
}


void init_neighbors() // create neighbor list
{
  // neighbor contains the index of the neighboring spin
  // for each spin there are four neighbors in this order: above, right, below, left
  neighbor = (int*) malloc(numberspins * numberneighbors * sizeof(int));

  for (int i=0; i<numberspins; i++)    // in general
    {
      neighbor[4*i]   = i - L1dim;     // above
      neighbor[4*i+1] = i + 1;         // right
      neighbor[4*i+2] = i + L1dim;     // below
      neighbor[4*i+3] = i - 1;         // left
    };

  if (bctype == 0)   // periodic BC
    for (int i=0; i<numberspins; i++)                  // now treat boundaries separately
      {
	if (i < L1dim)                                 // top row
	  neighbor[4*i] = numberspins - L1dim + i;
	if ((i+1)%L1dim == 0)                          // rightmost column
	  neighbor[4*i+1] = i + 1 - L1dim;
	if (i > (numberspins - L1dim - 1) )            // bottom row
	  neighbor[4*i+2] = i - (numberspins - L1dim);
	if (i%L1dim == 0)                              // leftmost column
	  neighbor[4*i+3] = i - 1 + L1dim;
      };

  if (bctype == 1) // Braskamp Kunz BC
    for (int i=0; i<numberspins; i++)                  // now treat boundaries separately
      {
	if ((i+1)%L1dim == 0)                          // rightmost column
	  neighbor[4*i+1] = i + 1 - L1dim;
	if (i%L1dim == 0)                              // leftmost column
	  neighbor[4*i+3] = i - 1 + L1dim;
	if (i < L1dim)                                 // top row
	  neighbor[4*i] = numberspins;
	if (i > (numberspins - L1dim - 1) )            // bottom row
	  neighbor[4*i+2] = numberspins + (i%2);
      };
  
  //print neighbor list (for manual debugging)
  //  for (int i=0; i<numberspins*numberneighbors; i++)
  //    {
  //      printf("%4d", neighbor[i]);
  //      if ((i+1)%numberneighbors == 0) printf("\n");
  //    };
}


int totalenergy()         // returns total energy of system
{
  int e = 0;
  for (int i=0; i<numberspins; i++)
    {
      if (bctype == 0)    // periodic boundaries
	for (int j=0; j<2; j++)
	  {
	    if (latticepoint[i] == latticepoint[neighbor[4*i+j]])
	      e--;
	  }
      if (bctype == 1)    // Braskamp Kunz boundaries
	for (int j=1; j<3; j++)
	  {
	    if (latticepoint[i] == latticepoint[neighbor[4*i+j]])
	      e--;
	  }
    };

  if (bctype == 1)        // Braskamp Kunz boundaries, add fixed upper boundaries
    for (int i=0; i<L1dim; i++)
      {
	if (latticepoint[i] == latticepoint[neighbor[4*i]])
	  e--;
      }
  return (e);
}


int local_energy(int i)   // returns energy of a single spin
{
  double eloc = 0;

  for(int j=0; j<numberneighbors; j++)
    {
      if (latticepoint[i] == latticepoint[neighbor[4*i+j]])
        eloc--;
    }
  return (eloc);
}


int propose_update(int w)    // returns energy change if a spin would be updated
{
  // ! this could be coded better: this function also changes the actual spin !
  // ! if update is not accepted, this has to be undone explicitly !
  int elocvor = local_energy(w);
  int newq = rand() % q;
  latticepoint[w] = newq;    // spin _is_ changed here

  int elocnach = local_energy(w);
  return(elocnach-elocvor);
}


int histflat(int imin, int imax, double ratio)
{
  // checks flatness of histograms for _all_ walkers in energy window
  int myflat, otherflat; 
  int flatness_crit = 1;

  // check own flatness first
  if (flatness_crit == 0)       // Zhou criterion
    {
      // NOT AVAILABLE YET!
      // TODO: CALCULATE histmin FOR ZHOU CRITERION
      //    myflat=1;
      //    for (int x=imin;x<=imax;x++)
      //    if (HE[x]<histmin) myflat=0;
    }
  else if (flatness_crit == 1)  // "Original" percentage criterion
    {
      myflat = 1;
      double minval;
      minval = HE[imin];        // take GS hits as reference
      double average = 0.0;
      double count = 0.0;
      
      for (int x=imin; x<=imax; x++)
	{
	  if (((x>5)||(x==4)) && (HE[x]<minval))
            minval = HE[x];
	  //(I am not sure right now why I included the first condition ...)

	  average += HE[x];
	  count++;
	}
      average /= count;
      
      if (minval < (ratio*average)) 
        myflat = 0;
    }

  // now talk to all the other walkers in the energy window
  // (! this whole thing can be reduced to an MPI_Allreduce once there 
  // are separate communicators for energy windows !)

  if (merge_hists == 1)                   // check flatness of other walkers in window
    {
      if (myid%multiple == 0)             // 'root' in energy window, receive individual flatnesses
	{
	  for (int i=1; i<multiple; i++)
	    {
	      MPI_Recv(&otherflat, 1, MPI_INT, myid+i, 66, MPI_COMM_WORLD, &status);
	      myflat *= otherflat;        // all have to be '1' to give '1' in the end (manual '&&')
	    }
	  for (int i=1; i<multiple; i++)  // and let everybody know
	    {
	      MPI_Send(&myflat, 1, MPI_INT, myid+i, 88, MPI_COMM_WORLD);
	    }
	}
      else                                // send individual flatness and receive 'merged' flatness
	{
	  MPI_Send(&myflat, 1, MPI_INT, myid-(myid%multiple), 66, MPI_COMM_WORLD);
	  MPI_Recv(&otherflat, 1, MPI_INT, myid-(myid%multiple), 88, MPI_COMM_WORLD, &status);
	  myflat = otherflat;             // replace individual flatness by merged
	}
    }
  return (myflat); 
  // note: by now, myflat refers to the 'collective' flatness in the energy window,
  // not the flatness of an individual walker
}


void init_lattice(double emin, double emax)
{
  int e,r;

  latticepoint = (int*) malloc((numberspins+2+1) * sizeof(int));
  // Note: we reserve space for 3 extra 'spins':
  // 2 extra to store fix values used for certain boundary conditions
  // 1 extra to carry an replica-id flag

  // find a fast way to create valid initial configurations
  // start at either Eglobalmin or Eglobalmax and change spins randomly
  // until energy is in the middle third of local energy range
  // global maximum of g(E) id at E/N=-0.2 

  //initialize lattice
  for (int i=0; i<numberspins; i++) 
    {
      if ((emin+(emax-emin)/2) > -0.2*(numberspins))   // start with ordered config at E=0
        latticepoint[i] = (i + ((i/L1dim) % 2)) % 2; 
      else                                             // start with GS config
        latticepoint[i] = 1;
    }

  e = totalenergy();

  stdoutlog = fopen(stdoutlogname, "a");
  fprintf(stdoutlog, "Proc. %i: Initialized lattice with energy e=%i, create setup with %lf<e<%lf\n", myid, e, (emin+(emax-emin)/3), (emin+2*(emax-emin)/3));
  fclose(stdoutlog);

  // run to a valid energy for the energy window

  // as long as energy is outside the middle third of local energy range
  while ((e < (emin+(emax-emin)/3)) || (e > (emin+2*(emax-emin)/3))) 
    {
      r = rand() % numberspins;
      e += propose_update(r);
    }

  if (bctype == 1)
    {
      latticepoint[numberspins] = 1;         // Braskamp Kunz has fixed spin values at boundaries
      latticepoint[numberspins+1] =- 1;
    }

  // use last spin to store replica-id; this is just a marker to
  // allows us to keep track of replicas as they are exchanged
  latticepoint[numberspins+2] = myid;

  //Print lattice to logfile
  stdoutlog = fopen(stdoutlogname, "a");
  for (int i=0; i<numberspins; i++) 
    {
      fprintf(stdoutlog, "%4i", latticepoint[i]);
      if ((i+1)%L1dim == 0)
        fprintf(stdoutlog, "\n");
    }
  fclose(stdoutlog);
}


void init_hists() // initialize histograms
{
  lngE     = (double*) malloc(hist_size*sizeof(double));
  lngE_buf = (double*) malloc(hist_size*sizeof(double));
  HE       = (double*) malloc(hist_size*sizeof(double));

  for (int i=0; i<hist_size; i++) 
    lngE[i] = 0.0;
}


// set boundaries for energy windows:
// either read from file (if precomputed for balanced REWL)
// or just split energy range evenly according to avaliable waklers
int find_local_energy_range(double Eglobmin, double Eglobmax, double overlap, int N)
{
  stdoutlog = fopen(stdoutlogname, "a");

  // consistency check
  if (N%multiple != 0)
    {
      fprintf(stdoutlog, "Total number of processes (%i) must be a multiple of the number of walkers per energy range (%i)!\n", N, multiple);
      fclose(stdoutlog);
      return (1);
    }

  FILE *window_init;

  // check if there is a file containing local energy window boundaries
  // Must be named Ewindows.dat !
  if ((window_init = fopen("Ewindows.dat","r")) == NULL)
    {
      fprintf(stdoutlog, "Proc %i: Can't find file Ewindows.dat. Will calculate equal-size windows with overlap %lf\n", myid, overlap);
      double Ewidth = (Eglobmax-Eglobmin)/(1.0 + ((double)(N/multiple)-1.0) * (1.0-overlap));

      Emin = Eglobmin + (double)(myid/multiple) * (1.0-overlap) * Ewidth;
      Emax = Emin + Ewidth;
      
      Eminindex = floor(Emin + 2*numberspins);
      Emaxindex = ceil(Emax + 2*numberspins);
      
      time(&timenow);
      fprintf(stdoutlog, "Proc %3i: Parameter: Eglobmin -- Eglobmax: %lf -- %lf; overlap=%i percent, Ewindowwidth=%lf, %s", myid, Eglobmin, Eglobmax, (int)(overlap*100.0), Ewidth, ctime(&timenow));
    }
  else
    {
      int myid_cmp;
      double E1tmp, E2tmp;
      fprintf(stdoutlog, "Proc %i: Read local energy boundaries from file Ewindows.dat\n", myid);
      for (int i=0; i<N/multiple; i++)
	{
	  if (fscanf(window_init, "%i\t%lf\t%lf\n", &myid_cmp, &E1tmp, &E2tmp) != EOF)
	    {
	      if (myid_cmp == myid/multiple)
		{
		  Eminindex = E1tmp;
		  Emaxindex = E2tmp;
		}
	    }
	  else
	    {
	      fprintf(stdoutlog, "Can't read boundaries for %ith energy window!\n", myid/multiple);
	      fclose(stdoutlog);
	      return (1);
	    }
	}
    }

  fclose(stdoutlog);

  return (0);
}


// THIS IS THE MASTER RE / SWAP FUNCTION
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
int replica_exchange(int swap_direction, int index_akt)
{
  int i_new; // histogram index of my configuration
  int Ecur; // current energy

  // frac refers to local exchange probability
  // wk is combined exchange probability
  double myfrac,otherfrac,randx,wk;

  int change=0; // boolean: 0 for not exchanging, 1 for exchanging
  int swap_partner=-1; // id of swap partner (receive_buffer)

  swap_direction=swap_direction%2; // comes actually as number of swap attempt
  
  // everyone has to find its swap-partner

  int *pairs; // array containing the partners of each process (send_buffer)
  pairs = (int*)malloc(2*multiple*sizeof(int));

  if (mylocalid[swap_direction]==0) // 'head-node' in the energy window determines pairs of flippartners
    {
      int choose_from=multiple; // number of free partners in higher window of communicator
      int select; // storage for random number
      
      int *libre; // list of free partners from higher window in communicator
      libre = (int*)malloc(multiple*sizeof(int));
      
      for (int i=0;i<multiple;i++) libre[i]=multiple+i; // initialise

      // idea: processes from the lower window choose someone from the higher window at random
      // of course, the chosen walker can't have an exchange partner yet
      for (int i=0;i<multiple;i++) // loop over processes in the lower window
	{
	  select=rand()%choose_from;
	  pairs[i]=libre[select];
	  pairs[libre[select]]=i; // the 'vice-versa pair'
	  // update list
	  choose_from--;
	  for (int j=select;j<choose_from;j++)
	    libre[j]=libre[j+1];
	}

//       stdoutlog=fopen(stdoutlogname,"a");
//       fprintf(stdoutlog,"Proc %3i: Drew the following swap partners:\n",myid);
//       for (int i=0;i<2*multiple;i++)
// 	fprintf(stdoutlog,"Proc %3i: %i -- %i (local ids in communicator)\n",myid,i,pairs[i]);
//       fclose(stdoutlog);
      
      free(libre);
    }

  // at this point, every walker has a swap partner assigned, now they must be communicated
  if ((swap_direction==0)&&(myid<(numprocs-multiple))) // the walkers from the last node should not swap
    {
      comm_id=2*(myid/(2*multiple)); // ! all integer, the '/' is a (div) ! Not the same as myid/multiple !
      MPI_Scatter(pairs,1,MPI_INT,&swap_partner,1,MPI_INT,0,mpi_local_comm[comm_id]);
    }

  if ((swap_direction==1)&&(myid>=multiple)) // the walkers from the zero-node should not swap
    {
      comm_id=((myid-multiple)/(2*multiple))*2+1; // ! all integer, the '/' is a (div) ! See above
      MPI_Scatter(pairs,1,MPI_INT,&swap_partner,1,MPI_INT,0,mpi_local_comm[comm_id]);
    }

  MPI_Barrier(MPI_COMM_WORLD);

  free(pairs);

  if (swap_partner!=-1) // i.e. if there is a swap-partner for me (if I am at a boundary, I might not have a swap partner this time)
    {
      // statistics
      if (swap_partner>mylocalid[swap_direction]) tryright++;
      else tryleft++;

      // safety cross check
      Ecur=totalenergy();
      if (Ecur+(2*numberspins)!=index_akt)
	{
	  stdoutlog=fopen(stdoutlogname,"a");
	  fprintf(stdoutlog,"Proc %3i, replica_exchange(): Something is wrong here! Received index=%i, calculated index=%i. Abort.\n",myid,index_akt,Ecur+(2*numberspins));
	  fclose(stdoutlog);
	  MPI_Abort(MPI_COMM_WORLD,1);
	}

      i_new=index_akt;

      // get histogram index from my swap partner
      MPI_Sendrecv_replace(&i_new,1,MPI_INT,swap_partner,1,swap_partner,1,mpi_local_comm[comm_id],&status);

      if ((i_new>Emaxindex)||(i_new<Eminindex)) // energyranges must overlap!
	{
	  myfrac=-1.0;
	}
      else
	{
	  // calculate my part of the exchange probability
	  myfrac=exp(lngE[index_akt]-lngE[i_new]); // g(myE)/g(otherE)
	}

      if (mylocalid[swap_direction]<multiple) // I am receiver and calculator
	{
	  // get my partners part of the exchange probability
	  MPI_Recv(&otherfrac,1,MPI_DOUBLE,swap_partner,2,mpi_local_comm[comm_id],&status);

	  // calculate combined exchange probability and roll the dice
	  if ((myfrac>0.0)&&(otherfrac>0.0))
	    {
	      randx=(1.0*rand()/(RAND_MAX+1.0));
	      wk=myfrac*otherfrac;
	      if (randx<wk) change=1;
	    }

	  // tell my swap partner whether to exchange or not
	  MPI_Send(&change,1,MPI_INT,swap_partner,3,mpi_local_comm[comm_id]);
	}
      else // I just send my part of exchange probability and await decision
	{
	  MPI_Send(&myfrac,1,MPI_DOUBLE,swap_partner,2,mpi_local_comm[comm_id]);
	  MPI_Recv(&change,1,MPI_INT,swap_partner,3,mpi_local_comm[comm_id],&status);
	}

      // if decision was made to exchange configurations
      if (change==1)
	{
	  // exchange spin conformations (incl. the 3 'special' lattice sites)
	  MPI_Sendrecv_replace(&latticepoint[0],numberspins+2+1,MPI_INT,swap_partner,1,swap_partner,1,mpi_local_comm[comm_id],&status);

	  // statistics
	  if (swap_partner>mylocalid[swap_direction]) exchangeright++;
	  else exchangeleft++;
	}
    }
  
  return(change);
  // returns whether or not configs were actually exchanged
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int main(int argc, char* argv[])
{
  // set up MPI and MPI_COMM_WORLD
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  // check command line arguments
  sprintf(stdoutlogname, "Error%04i.log", myid);
  if (argc < 4)
    {
      stdoutlog = fopen(stdoutlogname, "w");
      fprintf(stdoutlog, "Unexpected number of command line arguments!\n");
      fprintf(stdoutlog, "Expect 4 arguments, %d were provided.\n", argc);
      fprintf(stdoutlog, "Syntax: ./WLpotts_mpi [arg1] [arg2] [arg3] [arg4] \n\n");
      fprintf(stdoutlog, "Please provide the following command line arguments:\n");
      fprintf(stdoutlog, "1. Overlap between consecutive windows. [double, 0 <= overlap <= 1]\n");
      fprintf(stdoutlog, "2. Number of walkers per energy subwindow. [integer]\n");
      fprintf(stdoutlog, "3. Number of Monte Carlo steps between replica exchange. [integer]\n");
      fprintf(stdoutlog, "4. Random number seed. [integer]\n\n");
      fclose(stdoutlog);

      printf("ERROR: Unexpected number of command line arguments!\n");
      printf("       Expect 4 arguments, %d were provided.\n", argc);
      printf("Syntax: ./WLpotts_mpi [arg1] [arg2] [arg3] [arg4] \n\n");
      printf("Please provide the following command line arguments:\n");
      printf("1. Overlap between consecutive windows. [double, 0 <= overlap <= 1]\n");
      printf("2. Number of walkers per energy subwindow. [integer]\n");
      printf("3. Number of Monte Carlo steps between replica exchange. [integer]\n");
      printf("4. Random number seed. [integer]\n\n");

      MPI_Abort(MPI_COMM_WORLD, 1);
    }

  // number of walkers per energy window
  multiple = atoi(argv[2]);

  // at the moment, the code works only for an _odd_ number of energy windows
  // (to make the RE in windows at the outside consistent)
  if ((numprocs/multiple)%2 == 0)
    {
      stdoutlog = fopen(stdoutlogname, "a");
      fprintf(stdoutlog, "ERROR: Even number of energy windows (%d) requested. Please request an odd number of energy windows.\n\n", numprocs/multiple);
      fclose(stdoutlog);

      printf("ERROR: Even number of energy windows (%d) requested. Please request an odd number of energy windows.\n\n", numprocs/multiple);

      MPI_Abort(MPI_COMM_WORLD, 1);
    }

  //MPI_Barrier(MPI_COMM_WORLD);

  // initiate random numbers
  rseed = atoi(argv[4]);
  srand(rseed+myid);

  int swap_every = atoi(argv[3]);      // after this number of sweeps try conformations swap
  int swap_every_init = swap_every;

  // init log file
  sprintf(stdoutlogname, "Proc%04i.sim%i.log", myid, rseed);

  // set local energy range
  ret_status = find_local_energy_range(Eglobalmin, Eglobalmax, atof(argv[1]), numprocs);
  MPI_Barrier(MPI_COMM_WORLD);

  if (ret_status != 0)
    {
      stdoutlog = fopen(stdoutlogname, "a");
      fprintf(stdoutlog,"Proc. %3i: find_local_energy_range() returned %i\n", myid, ret_status);
      fclose(stdoutlog);

      MPI_Abort(MPI_COMM_WORLD, 1);    // something went wrong in find_local_energy_range()
    }

  init_neighbors();
  init_lattice(Emin,Emax); // 0 - random; 1 - all equal

  // calculate energy for the first time
  int eold,energie;
  energie=totalenergy();
  stdoutlog=fopen(stdoutlogname,"a");
  fprintf(stdoutlog,"Proc %3i: energy at start=%i\n",myid,energie);
  fclose(stdoutlog);
  energie-=Eglobalmin; // shift to positive values to use it as array index

  Estartindex=energie;

  int stop=0;
  stdoutlog=fopen(stdoutlogname,"a");
  fprintf(stdoutlog,"Proc %3i: Parameter: Eminindex -- Emaxindex: %i -- %i; Estartindex=%i\t",myid,Eminindex,Emaxindex,Estartindex);
  if ((Estartindex>Emaxindex)||(Estartindex<Eminindex))
    {
      fprintf(stdoutlog,"Estart out of range!!!\n");
      stop=1;
    }
  else
    {
      fprintf(stdoutlog,"OK\n");
    }

  fclose(stdoutlog);

  MPI_Barrier(MPI_COMM_WORLD);
  if (stop) MPI_Abort(MPI_COMM_WORLD,1);

//Teststop
//   MPI_Barrier(MPI_COMM_WORLD);
//   MPI_Abort(MPI_COMM_WORLD,1);

  // create new groups and communicators for each energy range window
  stdoutlog=fopen(stdoutlogname,"a");
  MPI_Comm_group(MPI_COMM_WORLD, &world); // get the group of processes in MPI_COMM_WORLD (i.e. all)
  int *ranks;
  ranks = (int*)malloc(2*multiple*sizeof(int));
  mpi_local_group = (MPI_Group*)malloc(((numprocs/multiple)-1)*sizeof(MPI_Group));
  mpi_local_comm = (MPI_Comm*)malloc(((numprocs/multiple)-1)*sizeof(MPI_Comm));

  for (int i=0;i<((numprocs/multiple)-1);i++) // i is counter for the energy range windows
    {
      for (int j=0;j<2*multiple;j++)
	{
	  ranks[j]=i*multiple+j; // contains the ranks of processes in MPI_COMM_WORLD which should get into local group
	  if (myid==0) 
	    {
	      fprintf(stdoutlog,"Proc %3i: %i will be part of communicator/group %i\n",myid,ranks[j],i);
	    }
	}
      MPI_Group_incl(world,2*multiple,ranks,&mpi_local_group[i]); // create local group
      MPI_Comm_create(MPI_COMM_WORLD,mpi_local_group[i],&mpi_local_comm[i]); // create communicator for that group
    }

  free(ranks);

  // get my local id (in my local communicators)
  if (myid<numprocs-multiple)
    {
      comm_id=2*(myid/(2*multiple));
      MPI_Comm_rank(mpi_local_comm[comm_id], &mylocalid[0]);
      fprintf(stdoutlog,"Proc %3i: I am part of communicator/group %i with local_id[0]=%i\n",myid,comm_id,mylocalid[0]);
    }
  else
    {
      mylocalid[0]=INT_MAX; // just to give it a value
      fprintf(stdoutlog,"Proc %3i: got local_id[0]=%i\n",myid,mylocalid[0]);
    }
  
  if (myid>=multiple)
    {
      comm_id=2*((myid-multiple)/(2*multiple))+1;
      MPI_Comm_rank(mpi_local_comm[comm_id], &mylocalid[1]);
      fprintf(stdoutlog,"Proc %3i: I am part of communicator/group %i with local_id[1]=%i\n",myid,comm_id,mylocalid[1]);
    }
  else
    {
      mylocalid[1]=INT_MAX; // just to give it a value
      fprintf(stdoutlog,"Proc %3i: got local_id[1]=%i\n",myid,mylocalid[1]);
    }


  // log 'path'data (the same data is written, just differently sorted: with respect to the sample id and with respect to process id)
  sprintf(wanderlogname,"wanderlog_rseed%i_sample_%i.dat",rseed,latticepoint[numberspins+2]);
  wanderlog=fopen(wanderlogname,"w");
  time(&timenow);
  fprintf(wanderlog,"#sweep\t\tproc-id\tconf-id\tenergy\t\ttime\n");
  fprintf(wanderlog,"%e\t%i\t%i\t%i\t%s",0.0,myid,latticepoint[numberspins+2],energie,ctime(&timenow));
  fclose(wanderlog);

  sprintf(wanderlogname,"wanderlog_rseed%i_walker_%i.dat",rseed,myid);
  wanderlog=fopen(wanderlogname,"w");
  time(&timenow);
  fprintf(wanderlog,"#sweep\t\tproc-id\tconf-id\tenergy\t\ttime\n");
  fprintf(wanderlog,"%e\t%i\t%i\t%i\t%s",0.0,myid,latticepoint[numberspins+2],energie,ctime(&timenow));
  fclose(wanderlog);

  //start simulation
  double wsk,dice; // wsk: probability  
  int wiggle; // ID of spin to be updated (only single-spin update implemented)

  init_hists();

  // the WL parameter should eventually go into the init file, too
  double lnf=1.0; // my modification factor
  double lnf_slowest=lnf; // modification factor of slowest walker in my window
  //double lnfmin=log(1.000000001);
  double lnfmin=log(1.0000001); // terminal modification factor
  double sweep=0; // counter for MC sweeps
  int flat; // 0 - histogram not flat; 1 - flat
  int iteration=1; // WL iteration counter
  double check_flatness_every=500; // in number of sweeps
  int backup;
  
  eold=energie;

  fprintf(stdoutlog,"Proc %3i: Start WL iteration\n",myid);
  fclose(stdoutlog);

  for(int i=0;i<=2*numberspins;i++) HE[i]=0; //init H(E)

  // Main Wang-Landau routine
  while (lnf_slowest>lnfmin)
    {
      for(int k=0;k<check_flatness_every;k++)
	{
	  for(int i=0;i<numberspins;i++) // this does 1 MC sweep
	    {
	      wiggle=rand()%numberspins; // choose spin to wiggle at at random
	      backup=latticepoint[wiggle]; // remember old spin orientation
	      
	      energie=eold+propose_update(wiggle); // calculate energy of updated configuration
	      
	      // reject because you got out of bounds of energy window
	      if ((energie>Emaxindex)||(energie<Eminindex)) 
		{
		  latticepoint[wiggle]=backup;
		  energie=eold;
		}
	      else // calculate acceptance propbability
		{
		  dice=(1.0*rand()/(RAND_MAX+1.0)); // roll the dice
		  wsk=exp(lngE[eold]-lngE[energie]); // calculate acceptance probability
		  if (dice>wsk)  // reject
		    {
		      latticepoint[wiggle]=backup;
		      energie=eold;
		    }
		  else eold=energie; // accept
		}
	      // update histograms
	      lngE[energie]+=lnf;
	      HE[energie]++;
	    } // end 1 sweep
	  sweep++; // sweep counter
	  
	  swap_every--; // RE counter (countdown)
	  if (swap_every==0) // ignition! (time to try to exchange configurations)
	    {
	      swap_every=swap_every_init; // reset countdown clock
	      if (replica_exchange(sweep/swap_every,eold)==1) // if configs were exchanged ('accept'-branch)
		{
		  energie=totalenergy()+2*numberspins; // calculate new Energyindex
		  eold=energie;
		}
	      // update histograms (independently of whether RE happened or not)
	      lngE[energie]+=lnf;
	      HE[energie]++;

	      // record position (process) of the actual sample
	      sprintf(wanderlogname,"wanderlog_rseed%i_sample_%i.dat",rseed,latticepoint[numberspins+2]);
	      wanderlog=fopen(wanderlogname,"a");
	      time(&timenow);
	      fprintf(wanderlog,"%e\t%i\t%i\t%i\t%s",sweep,myid,latticepoint[numberspins+2],eold,ctime(&timenow));
	      fclose(wanderlog);
	      
	      // record actual sample id for each walker
	      sprintf(wanderlogname,"wanderlog_rseed%i_walker_%i.dat",rseed,myid);
	      wanderlog=fopen(wanderlogname,"a");
	      time(&timenow);
	      fprintf(wanderlog,"%e\t%i\t%i\t%i\t%s",sweep,myid,latticepoint[numberspins+2],eold,ctime(&timenow));
	      fclose(wanderlog);
	    }
	} // end computation block
      
      // check flatness
      flat=histflat(Eminindex,Emaxindex,0.8);
      if (flat==0) // histogram not flat, do nothing except maybe some logging
	{
	  //stdoutlog=fopen(stdoutlogname,"a");
	  //fprintf(stdoutlog,"Proc %3i: %i iteration, %e sweeps, Histogram noch nicht flach\n",myid,iteration,sweep);
	  //fclose(stdoutlog);
	}
      else // histograms of all walkers are 'flat'
	{
	  if (lnf>lnfmin) // as long as modification factor is still larger than terminal value
	    {
	      sprintf(filename,"L%iq%i.HE.proc%04i.iter%i",L1dim,q,myid,iteration);
	      if ((file=fopen(filename,"w"))==NULL)
		{
		  stdoutlog=fopen(stdoutlogname,"a");
		  fprintf(stdoutlog,"Can not open file %s. Exit.\n",filename);
		  fclose(stdoutlog);
		  MPI_Abort(MPI_COMM_WORLD,1);
		}
	      else
		{
		  for (int i=Eminindex;i<=Emaxindex;i++) 
		    {
		      if (HE[i]>0.5) fprintf(file,"%i\t%i\t%e\t%e\t%e\n",i,(i-2*numberspins),0.0,lngE[i],HE[i]);
		    }
		  fclose(file);
		}
	    }
	  
	  // decrease modification factor
	  // canonical method (reduce by sqrt(2)) implemented
	  lnf/=2.0;

	  for(int i=0;i<=2*numberspins;i++) HE[i]=0; //reset H(E)
	  iteration++; // iteration counter

	  if (merge_hists==1) // merge g(E) estimators from multiple walkers in the same energy window
	    {
	      stdoutlog=fopen(stdoutlogname,"a");
	      if (myid%multiple==0) // 'root' in energy window, receive individual g(E) and send merged g(E)
		{
		  for (int i=1; i<multiple; i++)
		    {
		      MPI_Recv(&lngE_buf[0],hist_size,MPI_DOUBLE,myid+i,77,MPI_COMM_WORLD,&status); // get other dens. of states
		      fprintf(stdoutlog,"Proc %i: Received lngE from Proc. %i\n",myid,myid+i);
		      for (int j=0; j<hist_size; j++) lngE[j]+=lngE_buf[j]; // sum up for average
		    }
		  for (int j=0; j<hist_size; j++) lngE[j]/=(double)multiple; // normalize
		  for (int i=1; i<multiple; i++)
		    {
		      MPI_Send(&lngE[0],hist_size,MPI_DOUBLE,myid+i,99,MPI_COMM_WORLD);
		      fprintf(stdoutlog,"Proc %i: Sent merged lngE to Proc. %i\n",myid,myid+i);
		    }
		}
	      else // send individual g(E) and receive merged g(E)
		{
		  MPI_Send(&lngE[0],hist_size,MPI_DOUBLE,myid-(myid%multiple),77,MPI_COMM_WORLD);
		  fprintf(stdoutlog,"Proc %i: Sent lngE to Proc. %i\n",myid,myid-(myid%multiple));
		  MPI_Recv(&lngE_buf[0],hist_size,MPI_DOUBLE,myid-(myid%multiple),99,MPI_COMM_WORLD,&status);
		  fprintf(stdoutlog,"Proc %i: Received merged lngE from Proc. %i\n",myid,myid-(myid%multiple));
		  for (int j=0; j<hist_size; j++) lngE[j]=lngE_buf[j]; // replace individual lngE (could be done directly, yes)
		}
	      fclose(stdoutlog);
	    }
	}
      
      // check progress from all other windows
      MPI_Allreduce(&lnf,&lnf_slowest,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
      
      // just some logging
      if (flat==1)
	{
	  stdoutlog=fopen(stdoutlogname,"a");
	  fprintf(stdoutlog,"Proc %3i: Start %i iteration, %e sweeps total so far, lnf now %e (lnfmin=%.2e, lnf_slowest=%e)\n",myid,iteration,sweep,lnf,lnfmin,lnf_slowest);
	  fprintf(stdoutlog,"Proc %3i: tryleft: %i, exchangeleft %i (Akzeptanzleft:%.2lf) <--> tryright: %i, exchangeright %i (Akzeptanzright:%.2lf)\n",myid,tryleft,exchangeleft,(double)exchangeleft/(double)tryleft,tryright,exchangeright,(double)exchangeright/(double)tryright);
	  fclose(stdoutlog);
	}
      
    } // end while(lnf_slowest>lnfmin) -> this terminates the simulation
  
//   free(neighbor);
//   free(latticepoint);

  // normaliye results
  double norm=lngE[0]-log(q);
  for (int i=0;i<=4*numberspins;i++) lngE[i]-=norm;
  
  // write final data to file
  sprintf(filename,"L%iq%i.proc%04i.lngE",L1dim,q,myid);
  if ((file=fopen(filename,"w"))!=NULL)
    {
      for (int i=Eminindex;i<=Emaxindex;i++) 
	if (lngE[i]>0.5) fprintf(file,"%i\t%i\t%e\t%e\t%e\n",i,(i-2*numberspins),0.0,lngE[i],HE[i]);
      fclose(file);
    }

//   free(HE);
//   free(lngE);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  return(0); // you did it!
}
