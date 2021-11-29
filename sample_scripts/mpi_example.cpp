// This program shows off the basics of using MPI with C++
// By: Nick from CoffeeBeforeArch

#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <unistd.h>

#include "likwid.h"

using namespace std;


int compute(int n){

    int fk = 0;
    int fn = 1;
    int tmp;

    for(int i=0; i<n; i++){

        tmp = fn + fk;
        fk = fn;
        fn = tmp;
    }

    return fn;

 }

int main(int argc, char *argv[]) {
  // Unique rank is assigned to each process in a communicator
  int rank;

  // Total number of ranks
  int size;

  // The machine we are on
  char name[80];

  // Length of the machine name
  int length;

  // Initializes the MPI execution environment
  MPI_Init(&argc, &argv);

  // Get this process' rank (process within a communicator)
  // MPI_COMM_WORLD is the default communicator
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Get the total number ranks in this communicator
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Gets the name of the processor
  // Implementation specific (may be gethostname, uname, or sysinfo)
  MPI_Get_processor_name(name, &length);

  // Print out for each rank
  cout << "Hello, MPI! Rank: " << rank << " Total: " << size
       << " Machine: " << name << endl;

  double get_time = -omp_get_wtime();
 
  int threads;
  int tid; 
 
  #pragma omp parallel 
  //for(int i = 0; i < 4; ++i)
  {

  threads = omp_get_num_threads();
  tid = omp_get_thread_num();

  printf("1st layer, thread = %d out of %d threads\n", tid, threads);
  sleep(1);


  /*    #pragma omp parallel 
      {
        tid = omp_get_thread_num();
        int max_threads = omp_get_max_threads();
    int threads = omp_get_num_threads();
    printf("2nd layer, thread = %d out of %d threads\n", tid, threads);
          
          //sleep(2);

          // add computational task to see threads working
          int fn = compute(1000000000);
          //std::cout << "pi : " << pi << std::endl;


      }*/
  }

  get_time += omp_get_wtime();

  std::cout << "time passed : " << get_time << std::endl;

  // Terminate MPI execution environment
  MPI_Finalize();

  return 0;
}
