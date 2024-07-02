#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <iomanip>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <unsupported/Eigen/SparseExtra>   // includes saveMarket

#include <armadillo>
#include "generate_testMat_selInv.cpp"
#include "../read_write_functions.cpp"
#include "helper_functions.h"

#include "BTA.H"

using Eigen::VectorXd;
using Eigen::MatrixXd;

typedef Eigen::VectorXd Vect;

#define PRINT_MSG
//#define RECORD_TIMES

// ******************* 
#define DOUBLE_PREC
typedef double T;
#define assign_T(val) val
// ******************* 

/* ===================================================================== */

int main(int argc, char* argv[])
{

size_t i; // iteration variable
std::string valueType;

#ifdef DOUBLE_PREC
    printf("Template T is double.\n");
    valueType = "double";
#else
    printf("Invalid template T.\n Currently only double allowed.\n");
    exit(1);
#endif


#if 1 // dummy example

    int ns=3;
    int nss=0;
    int nt=4;
    int nb=2;
    int n = ns*nt + nb;

    SpMat Q       = gen_test_mat_base3(ns, nt, nb);

    Vect rhs = Vect::Random(n);
    //rhs.setOnes(n);

#else

    // TODO: optionally add input rhs
    if(argc != 1 + 5){
        std::cout << "wrong number of input parameters. " << std::endl;
        std::cerr << "BTA call : ns nt nss nb path/to/files solver_type" << std::endl;
        
        exit(1);
    }

    size_t ns  = atoi(argv[1]);
    size_t nt  = atoi(argv[2]);
    size_t nss = atoi(argv[3]);
    size_t nb  = atoi(argv[4]);
    size_t n   = ns*nt + nss + nb;

    std::string Q_file = argv[5];    
    file_exists(Q_file);
    SpMat Q      = read_sym_CSC(Q_file);
    //std::cout << "Q(1:15, 1:15) : \n" << Q.block(0,0,15,15) << std::endl;

    // initialize dummy rhs for testing
    Vect rhs = Vect::Random(n);

#endif // end dummy example or reading in matrices


    // =========================================================================== //
#if 1 // call eigen solver for comparison
    std::cout << "Call Eigen solver. " << std::endl;

    //SimplicialLLT<SpMat, Eigen::Lower, Eigen::NaturalOrdering<int>> solverQ;
    SimplicialLLT<SpMat> solverQ;
    solverQ.compute(Q);

   if(solverQ.info()!=Success) {
     cout << "Oh: Very bad" << endl;
   }

    Vect x_Eigen = solverQ.solve(rhs);
    std::cout << "norm(Q*x_Eigen - rhs) = " << (Q*x_Eigen - rhs).norm() << std::endl;

   SpMat L = solverQ.matrixL();
   if(n < 10){
        std:cout << "L: \n" << MatrixXd(L) << std::endl;
    }

   // compute log sum by hand
   double logDetEigen = 0.0;
   for(int i = 0; i<n; i++){
        logDetEigen += log(L.coeff(i,i));
   }
   logDetEigen *=2.0;
   std::cout << "logDetEigen = " << logDetEigen << std::endl;

    // fully invert matrix
    SpMat SpId(n,n);
    SpId.setIdentity();

    SpMat inv_Q_Eigen = solverQ.solve(SpId);

#endif

    // =========================================================================== //
    std::cout << "\nConverting Eigen Matrices to CSR format. " << std::endl;

    SpMat Q_lower = Q.triangularView<Lower>(); 
    // only take lower triangular part of Q
    size_t nnz    = Q_lower.nonZeros();

    printf("nnz(Q_lower) = %ld\n", nnz);
    size_t* ia; 
    size_t* ja;
    T* a; 
    T *b;
    T *x;

    // extract all inverse elements corresponding to nonzero entries in Q
    T* invQa;

    // extract only diagonal entries
    T *invDiag;

    b  = new T[n];
    x  = new T[n];

    // allocate memory
    ia = new long unsigned int [n+1];
    ja = new long unsigned int [nnz];
    a  = new T [nnz];

    invQa = new T[nnz];
    invDiag  = new T[n];

    Q_lower.makeCompressed();

    for (i = 0; i < n+1; ++i){
        ia[i] = Q_lower.outerIndexPtr()[i]; 
    }  

    for (i = 0; i < nnz; ++i){
        ja[i] = Q_lower.innerIndexPtr()[i];
    }  

    // cast as double or f
    for (i = 0; i < nnz; ++i){
        a[i] = (T) Q_lower.valuePtr()[i];
    }

    for(i = 0; i < n; i++){
        b[i] = (T) rhs[i];
    }

    // *** pin GPU & combine with appropriate cores *** //
    int GPU_rank = 0;
    cudaSetDevice(GPU_rank);
    int numa_node = topo_get_numNode(GPU_rank);

    int* hwt = NULL;
    int hwt_count = read_numa_threads(numa_node, &hwt);
    pin_hwthreads(1, &hwt[omp_get_thread_num()]);
    std::cout<<"Pinning GPU & hw threads. GPU rank : "<<GPU_rank <<", tid: "<<omp_get_thread_num()<<", NUMA domain ID: "<<numa_node;
    std::cout<<", hwthread: " << hwt[omp_get_thread_num()] << std::endl;
    // *********************************************** //

    printf("call BTA constructor.\n", nt); 
    BTA<T> *solver;
    solver = new BTA<T>(ns, nt, nss+nb, GPU_rank);

    int m = 2;
    Vect t_factorize_vec(m-1);
    T log_det;

    double t_factorise;
    double t_solve;

    double t_firstStageFactor;
    double t_secondStageForwardPass;
    double t_secondStageBackwardPass1;

    double flops_factorize;

    for(int iter=0; iter<m; iter++){
        printf("\niter = %d\n", iter);

        /*t_factorise = get_time(0.0);
        flops_factorize = solver->factorize_noCopyHost(ia, ja, a, log_det);
        t_factorise = get_time(t_factorise);
        printf("log det noCopyHost: %f\n", log_det);
        printf("time factorize noCopyHost: %f\n", t_factorise);*/

        t_factorise = get_time(0.0);
        flops_factorize = solver->factorize(ia, ja, a, t_firstStageFactor);
        log_det = solver->logDet(ia, ja, a);
        printf("logdet: %f\n", log_det);
        t_factorise = get_time(t_factorise);
        //printf("time factorize:             %f\n", t_factorise);

        t_solve = get_time(0.0); 
        double flops_solve = solver->solve(ia, ja, a, x, b, 1, t_secondStageForwardPass, t_secondStageBackwardPass1);
        t_solve = get_time(t_solve);
        //printf("flops solve:     %f\n", flops_solve);

        printf("time factorize:            %f\n",t_factorise);
        printf("time solve:                %f\n",t_solve);

        printf("Residual norm. :           %e\n", solver->residualNorm(x, b));
        printf("Residual norm normalized : %e\n", solver->residualNormNormalized(x, b));

#if 0   // Fused Factorize Solve
        double t_firstSecondStage;
        double t_secondStageBackwardPass2;

        T *x_new = new T[n];

        t_factorise = get_time(0.0);
        flops_factorize = solver->factorizeSolve(ia, ja, a, x_new, b, 1, t_firstSecondStage, t_secondStageBackwardPass2);
        t_factorise = get_time(t_factorise);
        log_det = solver->logDet(ia, ja, a);

        Vect x_new_vec(n);
        Vect x_vec(n);

        for(int i=0; i<n; i++){
            x_new_vec[i] = x_new[i];
            x_vec[i]     = x[i];
        }
        std::cout << "norm(x-x_new) = " << (x_vec - x_new_vec).norm() << std::endl;

        printf("log det factorizeSolve   : %f\n", log_det);
        printf("time factorizeSolve      : %f\n", t_factorise);
#endif

    // Selected Inversion

    // extract only the diagonal entries
    double t_invDiag;
    t_invDiag = get_time(0.0);
    double flops_invDiag = solver->BTAdiag(ia, ja, a, invDiag);
    t_invDiag = get_time(t_invDiag);
    printf("time BTAdiag: %f\n", t_invDiag);
    double log_detBTAdiag = solver->logDet(ia, ja, a);

    if(n < 10){
        printf("\nBTAinvDiag: ");
        for(i=0; i<n; i++){
            printf(" %f", invDiag[i]);
        }
        printf("\n");
    }

    t_invDiag = get_time(0.0);
    double flops_invQa = solver->BTAselInv(ia, ja, a, invQa);
    t_invDiag = get_time(t_invDiag);
    printf("time BTAselInv: %f\n", t_invDiag);

    if(n < 10){
        printf("invQa : ");
        for(int i=0; i<nnz; i++){
            printf(" %f", invQa[i]);
        }
        printf("\n");
    }

    // store in matrix
    SpMat invQ_new_lower = Eigen::Map<Eigen::SparseMatrix<double> >(n,n,nnz,Q_lower.outerIndexPtr(), // read-write
                                Q_lower.innerIndexPtr(),invQa);

    if(n < 10){
        std::cout << "invQ_new:\n" << MatrixXd(invQ_new_lower) << std::endl;
    }

    // TODO: more efficient way to do this?
    SpMat invQ_new = invQ_new_lower.selfadjointView<Lower>();

    Vect invDiag_vec(n);
    for(int i=0; i<n; i++){
        invDiag_vec[i] = invDiag[i];
    }

    std::cout << "norm(diag(invQ_new) - diag(invDiag)) = " << (invQ_new.diagonal() - invDiag_vec).norm() << std::endl;
    std::cout << "norm(diag(invQ_new) - diag(invEigen)) = " << (invQ_new.diagonal() - inv_Q_Eigen.diagonal()).norm() << std::endl;

  delete[] invDiag;
  delete[] invQa;

    }
  
  // free memory
  delete solver;

  delete[] ia;
  delete[] ja;
  delete[] a;

  delete[] x;
  delete[] b;

  return 0;
  }

