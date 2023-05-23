#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>


// choose one of the two
#define DATA_SYNTHETIC
//#define DATA_TEMPERATURE

// enable RGF solver or not
#define RGF_SOLVER

#ifdef RGF_SOLVER
#include "cuda_runtime_api.h" // to use cudaGetDeviceCount()
#endif

//#define WRITE_RESULTS

// if predict defined -> account for missing data
#define PREDICT    

//#define PRINT_MSG
//#define WRITE_LOG

#include "mpi.h"

//#include <likwid.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <Eigen/SparseCholesky>


#include <armadillo>
#include <LBFGS.h>

#include "PostTheta.h"
#include "../read_write_functions.cpp"

// comment out when not needed
#include "generate_regression_data.cpp"

using Eigen::MatrixXd;
typedef Eigen::VectorXd Vect;
typedef Eigen::SparseMatrix<double, RowMajor> SpRmMat;

using namespace LBFGSpp;


/*void create_validation_set(int& no, int& size_valSet, std::vector<int> &indexSet, std::vector<int> &valSet){

    //int no = 30;
    //int size_valSet = 8;

    // requires C++-17 !!
    // create sorted index vector
    std::mt19937 rbg { 42u }; 

    //std::vector<int> indexSet(no);
    std::iota(indexSet.begin(), indexSet.end(), 0);
    //std::vector<int> valSet(size_valSet);
    // sample random indices
    std::sample(indexSet.begin(), indexSet.end(), valSet.begin(), valSet.size(), rbg);
    
    for (int valIndex: indexSet) std::cout << valIndex << ' '; 
    std::cout << '\n';

    for (int valIndex: valSet) std::cout << valIndex << ' '; 
    std::cout << '\n';

    // assuming sorted vectors : removes all elements of valSet that are in indexSet
    indexSet.erase( remove_if( begin(indexSet),end(indexSet),
    [&valSet](auto x){return binary_search(begin(valSet),end(valSet),x);}), end(indexSet) );

    for( int elem: valSet) std::cout << elem << ' '; 
    std::cout << '\n';

    for( int elem: indexSet) std::cout << elem << ' '; 
    std::cout << '\n';
}*/

// construct sparse Matrix from invBlks 
// invBlks have particular order ... non-contiguous ... 
// but we want to fill sparse matrix by column 
void construct_lower_CSC_invBlks(size_t ns, size_t nt, size_t nb, size_t nnz_lower_invBlks, double* invBlks, SpMat& QinvBlks){

    //std::cout << "nnz_invBlks lower = " << nnz_lower_invBlks << ", ns = " << ns << ", nt = " << nt << ", nb = " << nb << std::endl;
    int n = ns*nt + nb;
    SpMat QinvBlks_lower(n,n);
    QinvBlks_lower.reserve(nnz_lower_invBlks);

    int* row_ind_a; // row index of each nnz value
    int* col_ptr_a; // list of val indices where each column starts
    double* a;

    row_ind_a = new int [nnz_lower_invBlks];
    col_ptr_a = new int [n+1];
    a         = new double[nnz_lower_invBlks];

    size_t counter = 0;

    double t_loop = - omp_get_wtime();

    // deal with final block later
    // only read-out values from lower triangular part
    for(int col=0; col<ns*nt; col++){
        int ts = col / ns; // determine which time step we are in
        int col_ts = col % ns; // col relative to the current timestep
        //printf("ts = %d, col ts = %d, counter = %ld\n", ts, col_ts, counter);
        
        col_ptr_a[col] = counter;

        for(int i=col_ts; i<ns; i++){
            int row = ns * ts + i;
            // get correct entry from invBlks array
            // invBlks sorted as D1, F1, D2, F2, ... Fn, Dn+1n+1, column-major
            int ind_invBlks = ts * (ns + nb) * ns + col_ts * ns + i;
            //QinvBlks_lower.insert(row, col) = invBlks[ind_invBlks];
            a[counter]         = invBlks[ind_invBlks];
            row_ind_a[counter] = row;
            counter++;

        }

        for(int i=0; i<nb; i++){   
            // always last rows
            int row = ns*nt+i;
            int ind_invBlks = ts * (ns + nb) * ns + ns * ns + col_ts * nb + i;

            a[counter] = invBlks[ind_invBlks];
            row_ind_a[counter] = row;
            counter++;

        }
    }

    // ... and finally for last nb columns
    int ind_offset = (ns + nb) * ns * nt;
    for(int col=ns*nt; col<ns*nt+nb; col++){
        int col_ts = col % ns;
        //printf("col ts = %d\n", col_ts);

        col_ptr_a[col] = counter;

        for(int i=col_ts; i<nb; i++){
            int row = ns*nt+i;
            int ind_invBlks = ind_offset + col_ts*nb + i;

            a[counter] = invBlks[ind_invBlks];
            row_ind_a[counter] = row;
            counter++;
        }
    }

    // final column ptr entry
    col_ptr_a[n] = counter;

    t_loop += omp_get_wtime();
    printf("lowerCSC: time in loop : %f\n", t_loop);

#if 0
    printf("row_ind_a: ");
    for(int i=0; i<nnz_lower_invBlks; i++){
        printf("%d ", row_ind_a[i]);
    }
    printf("\n");


    printf("col_ptr_a: ");
    for(int i=0; i<n+1; i++){
        printf("%d ", col_ptr_a[i]);
    }
    printf("\n");

    printf("a: ");
    for(int i=0; i<nnz_lower_invBlks; i++){
        printf("%f ", a[i]);
    }
    printf("\n");
#endif

    t_loop = - omp_get_wtime();
    QinvBlks_lower =  Eigen::Map<Eigen::SparseMatrix<double> >(n,n,nnz_lower_invBlks,col_ptr_a, row_ind_a,a);
    t_loop += omp_get_wtime();
    printf("lowerCSC: time Eigen Map : %f\n", t_loop);

    t_loop = - omp_get_wtime();
    QinvBlks = QinvBlks_lower.selfadjointView<Lower>();
    t_loop += omp_get_wtime();
    printf("lowerCSC: time Eigen copy symmetrize : %f\n", t_loop);
    //std::cout << "QinvBlks : \n" << MatrixXd(QinvBlks) << std::endl;

}


// construct sparse Matrix from invBlks -> generate full CSC structure -> no symmetric multiplication routine
// invBlks have particular order ... non-contiguous ... 
// but we want to fill sparse matrix by column
void construct_full_CSC_invBlks(size_t ns, size_t nt, size_t nb, size_t nnz_invBlks, double* invBlks, SpMat& QinvBlks){

    //std::cout << "nnz_invBlks lower = " << nnz_lower_invBlks << ", ns = " << ns << ", nt = " << nt << ", nb = " << nb << std::endl;
    int n = ns*nt + nb;
    //SpMat QinvBlks_lower(n,n);
    QinvBlks.reserve(nnz_invBlks);

    int* row_ind_a; // row index of each nnz value
    int* col_ptr_a; // list of val indices where each column starts
    double* a;

    row_ind_a = new int [nnz_invBlks];
    col_ptr_a = new int [n+1];
    a         = new double[nnz_invBlks];

    size_t counter = 0;

    double t_loop = - omp_get_wtime();

    // deal with final block later
    // only read-out values from lower triangular part
    for(int col=0; col<ns*nt; col++){
        int ts = col / ns; // determine which time step we are in
        int col_ts = col % ns; // col relative to the current timestep
        //printf("ts = %d, col ts = %d, counter = %ld\n", ts, col_ts, counter);
        
        col_ptr_a[col] = counter;

        for(int i=0; i<ns; i++){
            int row = ns * ts + i;
            // get correct entry from invBlks array
            // invBlks sorted as D1, F1, D2, F2, ... Fn, Dn+1n+1, column-major
            int ind_invBlks = ts * (ns + nb) * ns + col_ts * ns + i;
            //QinvBlks_lower.insert(row, col) = invBlks[ind_invBlks];
            a[counter]         = invBlks[ind_invBlks];
            row_ind_a[counter] = row;
            counter++;

        }

        for(int i=0; i<nb; i++){   
            // always last rows
            int row = ns*nt+i;
            int ind_invBlks = ts * (ns + nb) * ns + ns * ns + col_ts * nb + i;

            a[counter] = invBlks[ind_invBlks];
            row_ind_a[counter] = row;
            counter++;

        }
    }

    int ind_offset = (ns + nb) * ns * nt;
    for(int col=ns*nt; col<ns*nt+nb; col++){
        int col_ts = col % ns;
        //printf("col ts = %d\n", col_ts);

        col_ptr_a[col] = counter;

        for(int ts=0;ts<nt; ts++){
            for(int ss=0; ss<ns; ss++){
                int row = ts*ns + ss;
                int ind_invBlks = (ns+nb)*ns*ts + ns*ns + col_ts + ss*nb;
                a[counter] = invBlks[ind_invBlks];
                row_ind_a[counter] = row;
                counter++;
            }

        }

        for(int i=0; i<nb; i++){
            int row = ns*nt+i;
            int ind_invBlks = ind_offset + col_ts*nb + i;

            a[counter] = invBlks[ind_invBlks];
            row_ind_a[counter] = row;
            counter++;
        }
    }

    // final column ptr entry
    col_ptr_a[n] = counter;

    t_loop += omp_get_wtime();
    printf("fullCSC: time in loop : %f\n", t_loop);

    t_loop = - omp_get_wtime();
    QinvBlks =  Eigen::Map<Eigen::SparseMatrix<double> >(n,n,nnz_invBlks,col_ptr_a, row_ind_a,a);
    t_loop += omp_get_wtime();
    printf("fullCSC: time Eigen Map : %f\n", t_loop);

    //std::cout << "QinvBlks full: \n" << MatrixXd(QinvBlks) << std::endl;

}


// construct sparse Matrix from invBlks -> generate full CSC structure -> no symmetric multiplication routine
// invBlks have particular order ... non-contiguous ... 
// but we want to fill sparse matrix by column
void get_projMargVar(size_t ns, size_t nt, size_t nb, size_t nnz_invBlks, double* invBlks, SpRmMat& A, Vect& projMargVar){

    // for block array returned from GPU ...
    size_t nrows = A.rows();
    size_t ncols = A.cols();

    size_t n = ns*nt+nb; 

    if(n != ncols){
        printf("Dimensions don't match! ns*nt + nb = %ld, cols(A) = %ld\n", n, ncols);
        exit(1);
    }

    SpMat QinvBlks;
    //size_t nnz_lower_invBlks = nt*ns*(ns+1)/2 + ns*nb*nt + (nb+1)*nb/2;
    //construct_lower_CSC_invBlks(ns, nt, nb, nnz_lower_invBlks, invBlks, QinvBlks);
    construct_full_CSC_invBlks(ns, nt, nb, nnz_invBlks, invBlks, QinvBlks);
    //std::cout << "QinvBlks:\n" << MatrixXd(QinvBlks) << std::endl;

    projMargVar.setZero();

    Vect F_comp(nrows);
    F_comp.setZero();

    double t_AinvQAt = -omp_get_wtime();

    for (int row = 0; row < nrows; row++){
        //printf("\nrow = %d\n", row);
        int arow = row;  
        Eigen::SparseVector<double> T_vect(ncols);
        //Eigen::SparseVector<double> T_comp(ncols);
        T_vect.setZero();
        //T_comp.setZero();
        // we only have to iterate through the columns of S, where A(row,col) != 0
        for(int sind = A.outerIndexPtr()[row]; sind < A.outerIndexPtr()[row+1]; sind++){
            int scol = A.innerIndexPtr()[sind];  
            int ts = scol / ns; 
            int col_rel = scol % ns;
            // iterating through non-zero entries of each row of A
            T_vect.insert(scol) =0.0;
            //T_comp.insert(scol) =0.0;
            for(int aind = A.outerIndexPtr()[row]; aind < A.outerIndexPtr()[row+1]; aind++){   
                int acol = A.innerIndexPtr()[aind]; 
                int acol_shift = acol % ns;
                int acol_ts = acol / ns; // determine which time step we are in
                
                //printf("A: row = %d, col = %d , value = %f\n", row, A.innerIndexPtr()[aind], A.valuePtr()[aind]);
                //printf("S(%d, %d) = %f\n", acol, scol, S.coeff(acol, scol));
                // to compute T(row, col) += A(row, acol)*S(acol, col)

                if((acol_ts == ts) && (acol < ns*nt) && (scol < ns*nt)){
                    // acol is the desired row index
                    int ind_invBlks = ts * (ns + nb) * ns + col_rel * ns + acol_shift;
                    double invBlkVal = invBlks[ind_invBlks];
                    //printf("ts = %d, col_rel = %d, scol = %d, acol = %d\n", ts, col_rel, scol, acol);
                    //printf("A(%d, %d) = %f, \n", row, acol, A.valuePtr()[aind]);
                    //printf("ind_invBlks = %d, QinvBlk(%d, %d) = %f, SHOULD BE: %f\n", ind_invBlks, acol, scol, invBlkVal, QinvBlks.coeffRef(acol, scol));
                    T_vect.coeffRef(scol) += A.valuePtr()[aind]*invBlks[ind_invBlks];

                }
                if(acol < ns*nt && scol >= ns*nt) {
                // to compute T(scol) += A(row, acol)*S(acol, scol)
                    int ind_invBlks = (ns+nb)*ns*acol_ts + ns*ns + acol_shift*nb + col_rel;
                    double invBlkVal = invBlks[ind_invBlks];
                    //printf("ts = %d, col_rel = %d, scol = %d, acol = %d\n", ts, col_rel, scol, acol);
                    //printf("ind_invBlks = %d, QinvBlk(%d, %d) = %f, SHOULD BE: %f\n", ind_invBlks, acol, scol, invBlkVal, QinvBlks.coeffRef(acol, scol));
                    T_vect.coeffRef(scol) += A.valuePtr()[aind]*invBlks[ind_invBlks];
                }

                if(acol >= ns*nt && scol < ns*nt) {
                // to compute T(scol) += A(row, acol)*S(acol, scol)
                    int ind_invBlks = ts * (ns + nb) * ns + ns * ns + col_rel * nb + acol_shift;
                    double invBlkVal = invBlks[ind_invBlks];
                    //printf("ts = %d, col_rel = %d, scol = %d, acol = %d\n", ts, col_rel, scol, acol);
                    //printf("ind_invBlks = %d, QinvBlk(%d, %d) = %f, SHOULD BE: %f\n", ind_invBlks, acol, scol, invBlkVal, QinvBlks.coeffRef(acol, scol));
                    T_vect.coeffRef(scol) += A.valuePtr()[aind]*invBlks[ind_invBlks];

                }
                if(acol >= ns*nt && scol >= ns*nt) {
                    int ind_offset = (ns + nb) * ns * nt;
                    int ind_invBlks = ind_offset + col_rel*nb + acol_shift;
                    double invBlkVal = invBlks[ind_invBlks];
                    //printf("ind_invBlks = %d, QinvBlk(%d, %d) = %f, SHOULD BE: %f\n", ind_invBlks, acol, scol, invBlkVal, QinvBlks.coeffRef(acol, scol));
                    T_vect.coeffRef(scol) += A.valuePtr()[aind]*invBlks[ind_invBlks];
                }
                //T_comp.coeffRef(scol) += A.valuePtr()[aind]*QinvBlks.coeff(acol, scol);

            }
        }

        //std::cout << "row = " << row << ", T_vect : " << T_vect.transpose();
        //std::cout << "row = " << row << ", T_comp : " << T_comp.transpose() << std::endl;

        // now multiply by A^T, 1 scalarproduct in each row, iterate again for outerindexPtr of A  
        for(int aind = A.outerIndexPtr()[row]; aind < A.outerIndexPtr()[row+1]; aind++){   
            int acol = A.innerIndexPtr()[aind]; 
            //printf("A: row = %d, col = %d , value = %f\n", row, acol, A.valuePtr()[aind]);
            //printf("T(%d) = %f\n", acol, T_vect.coeffRef(acol));
            //printf("A.valuePtr()[aind]*T_vect.coeffRef(acol) = %f\n", A.valuePtr()[aind]*T_vect.coeffRef(acol));
            projMargVar(row) += A.valuePtr()[aind]*T_vect.coeffRef(acol);
            //F_comp(row) += A.valuePtr()[aind]*T_comp.coeffRef(acol);
            //printf("F(%d) = %f\n", row, F_vect(row));
        }

    }  

    t_AinvQAt += omp_get_wtime();
    printf("time AinvQAt : %f\n", t_AinvQAt);

    SpMat F_true(nrows,nrows);
    F_true = A*QinvBlks*A.transpose();

    //std::cout << "norm(F_comp - F_vect) = " << (F_comp - projMargVar).norm() << std::endl;
    std::cout << "norm(F_vect - F_true) = " << (F_true.diagonal() - projMargVar).norm() << std::endl;

    //std::cout << "F_comp - F_vect       = " << F_comp.transpose() - projMargVar.transpose() << std::endl;
    //std::cout << "F_comp - diag(F_true) = " << F_comp.transpose() - F_true.diagonal().transpose() << std::endl;


//std::cout << "\ndiag(F)" << F_vect.transpose() << std::endl;
//std::cout << "F_true : " << F_true.diagonal().transpose() << std::endl;

}


#if 0

// ============================================================================== //
// for block array returned from GPU ...

for (int row = 0; row < nrows; row++){
    //printf("\nrow = %d\n", row);
    int arow = row;  
    Eigen::SparseVector<double> T_vect(ncols);
    T_vect.setZero();
    // we only have to iterate through the columns of S, where A(row,col) != 0
    //for(int scol = 0; scol < ncols; scol++){
    for(int sind = A.outerIndexPtr()[row]; sind < A.outerIndexPtr()[row+1]; sind++){
        int scol = A.innerIndexPtr()[sind];    
        // iterating through non-zero entries of each row of A
        T_vect.insert(scol) =0.0;
        for(int aind = A.outerIndexPtr()[row]; aind < A.outerIndexPtr()[row+1]; aind++){   
            int acol = A.innerIndexPtr()[aind]; 
            //printf("A: row = %d, col = %d , value = %f\n", row, A.innerIndexPtr()[aind], A.valuePtr()[aind]);
            //printf("S(%d, %d) = %f\n", acol, scol, S.coeff(acol, scol));
            // entry of S: S
            // to compute T(row, col) += A(row, acol)*S(acol, col)
            T_vect.coeffRef(scol) += A.valuePtr()[aind]*S.coeff(acol, scol);
        }
    }
    //std::cout << "T.row(" << row << ") : " << T_vect.transpose() << std::endl;  
    // now multiply by A^T, 1 scalarproduct in each row, iterate again for outerindexPtr of A  
    for(int aind = A.outerIndexPtr()[row]; aind < A.outerIndexPtr()[row+1]; aind++){   
        int acol = A.innerIndexPtr()[aind]; 
        //printf("A: row = %d, col = %d , value = %f\n", row, acol, A.valuePtr()[aind]);
        //printf("T(%d) = %f\n", acol, T_vect.coeffRef(acol));
        //printf("A.valuePtr()[aind]*T_vect.coeffRef(acol) = %f\n", A.valuePtr()[aind]*T_vect.coeffRef(acol));
        F_vect(row) += A.valuePtr()[aind]*T_vect.coeffRef(acol);
        //printf("F(%d) = %f\n", row, F_vect(row));
    }

}  

#endif



void construct_Q_spat_temp(Vect& theta, SpMat& c0, SpMat& g1, SpMat& g2, SpMat& g3, SpMat& M0, SpMat& M1, SpMat& M2, SpMat& Qst){

    double exp_theta1 = exp(theta[1]);
    double exp_theta2 = exp(theta[2]);
    double exp_theta3 = exp(theta[3]);

    // g^2 * fem$c0 + fem$g1
    SpMat q1s = pow(exp_theta2, 2) * c0 + g1;

    // g^4 * fem$c0 + 2 * g^2 * fem$g1 + fem$g2
    SpMat q2s = pow(exp_theta2, 4) * c0 + 2 * pow(exp_theta2,2) * g1 + g2;

    // g^6 * fem$c0 + 3 * g^4 * fem$g1 + 3 * g^2 * fem$g2 + fem$g3
    SpMat q3s = pow(exp_theta2, 6) * c0 + 3 * pow(exp_theta2,4) * g1 + 3 * pow(exp_theta2,2) * g2 + g3;

    // assemble overall precision matrix Q.st
    Qst = pow(exp_theta1,2)*(KroneckerProductSparse<SpMat, SpMat>(M0, q3s) + 2*exp_theta3 *KroneckerProductSparse<SpMat, SpMat>(M1, q2s) + pow(exp_theta3, 2)* KroneckerProductSparse<SpMat, SpMat>(M2, q1s));

}

/* ===================================================================== */

int main(int argc, char* argv[])
{
    // start timer for overall runtime
    double t_total = -omp_get_wtime();

    /* ======================= SET UP MPI ============================= */
    // Unique rank is assigned to each process in a communicator
    int MPI_rank;

    // Total number of ranks
    int MPI_size;

    // Initializes the MPI execution environment
    MPI_Init(&argc, &argv);

    // Get this process' rank (process within a communicator)
    // MPI_COMM_WORLD is the default communicator
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);

    // Get the total number ranks in this communicator
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_size); 

    int threads_level1;
    int threads_level2;

    if(omp_get_nested() == true){
	threads_level1 = omp_get_max_threads();
	#pragma omp parallel
	{
	    threads_level2 = omp_get_max_threads();
	}
    } else {
	    //threads_level1 = omp_get_max_threads();
	    //threads_level2 = omp_get_max_threads();
    	threads_level2 = 1;
    }

    // overwrite in case RGF is used
    int noGPUs;
   
    if(MPI_rank == 0){
        printf("\n============== PARALLELISM & NUMERICAL SOLVERS ==============\n");
        printf("total no MPI ranks  : %d\n", MPI_size);
        printf("OMP threads level 1 : %d\n", threads_level1);
        printf("OMP threads level 2 : %d\n", threads_level2);
	    //printf("OMP threads level 2 FIXED TO 1!!\n");
#ifdef RGF_SOLVER
	cudaGetDeviceCount(&noGPUs);
	printf("available GPUs      : %d\n", noGPUs);
#else
	printf("RGF dummy version\n");
    noGPUs = 0;
#endif
    }  

    
    if(argc != 1 + 8 && MPI_rank == 0){
        std::cout << "wrong number of input parameters. " << std::endl;

        std::cerr << "INLA Call : ns nt nb t_fit path/to/files solver_type" << std::endl;

        std::cerr << "[integer:ns]                number of spatial grid points " << std::endl;
        std::cerr << "[integer:nb]                number of fixed effects" << std::endl;

        std::cerr << "[integer:nt_fit]            number of days used for fitting" << std::endl;
        std::cerr << "[integer:nt_pred]           number of days predicted" << std::endl;
        std::cerr << "[integer:nt_total]          number of days for which we have data" << std::endl;

        std::cerr << "[integer:no_per_ts]         number of data samples per ts (includes NA)" << std::endl;

        std::cerr << "[string:base_path]          path to folder containing matrix files " << std::endl;
        std::cerr << "[string:solver_type]        BTA or PARDISO " << std::endl;

        exit(1);
    }

#ifdef PRINT_MSG
    if(MPI_rank == 0){
        std::cout << "reading in example. " << std::endl;
    }
#endif

    size_t ns        = atoi(argv[1]);
    size_t nb        = atoi(argv[2]);

    size_t nt_fit    = atoi(argv[3]);
    size_t nt_pred   = atoi(argv[4]);
    size_t nt        = nt_fit + nt_pred; // internal model size
    size_t nt_total  = atoi(argv[5]);

    size_t no_per_ts = atoi(argv[6]);
    size_t no        = nt*no_per_ts;

    // set nt = 1 if ns > 0 & nt = 0
    if(ns > 0 && nt == 0){
        nt = 1;
    } 

    size_t n = ns*(nt_fit + nt_pred) + nb;

    // also save as string
    std::string ns_s        = std::to_string(ns);
    std::string nt_s        = std::to_string(nt);    
    std::string nb_s        = std::to_string(nb);

    std::string nt_fit_s    = std::to_string(nt_fit);
    std::string nt_pred_s   = std::to_string(nt_pred);
    std::string nt_total_s  = std::to_string(nt_total);

    std::string no_per_ts_s = std::to_string(no_per_ts); 
    std::string n_s         = std::to_string(n);

    std::string base_path   = argv[7];    
    std::string solver_type = argv[8];

    // check if solver type is neither PARDISO nor RGF :
    if(solver_type.compare("PARDISO") != 0 && solver_type.compare("BTA") != 0){
        std::cout << "Unknown solver type. Available options are :\nPARDISO\nBTA" << std::endl;
        exit(1);
    }

    if(MPI_rank == 0){
        std::cout << "Solver : " << solver_type << std::endl;
    }

    // we have two cholesky factors ...
    if(MPI_rank == 0 && solver_type.compare("BTA") == 0){
        // required memory on CPU to store Cholesky factor
        double mem_gb = (2*(nt-1)*ns*ns + ns*ns + (ns*nt+nb)*nb) * sizeof(double) / pow(10.0,9.0);
        printf("Memory Usage of each Cholesky factor on CPU = %f GB\n\n", mem_gb);
    }


// implement sparse multiplication of A*BlkDiag(Q^-1)*A*T
#if 0

size_t ns = 100;
size_t nt = 30;
size_t nb = 4;
size_t n = ns*nt+nb;

size_t nnz_invBlks = ns*ns*nt+2*nb*ns*nt + nb*nb;
// generate random Vector nnz_invBlks entries
Vect invBlks = Vect::Random(nnz_invBlks);
//std::cout << "invBlks : " << invBlks.transpose() << std::endl;

/*
SpMat QinvBlks(n,n);
construct_full_CSC_invBlks(ns, nt, nb, nnz_invBlks, invBlks.data(), QinvBlks);
std::cout << "QinvBlks:\n" << MatrixXd(QinvBlks) << std::endl;
*/

size_t no = ns*nt+nb+3;    

int nrows = no;
int ncols = n;

// generate sparse matrix A with random sparsity pattern
MatrixXd A_dense = MatrixXd::Random(nrows,ncols);

//std::cout << "A: \n" << A_dense << std::endl;


 for(int i=0; i<nrows; i++){
    for(int j=0; j<ncols; j++){
        if(A_dense(i,j) < 0.4){
            A_dense(i,j) = 0;
        }
    }
 }

 //std::cout << "A: \n" << A_dense << std::endl;

 MatrixXd S_dense =  MatrixXd::Random(ncols,ncols);

 //std::cout << "S: \n" << S_dense << std::endl;

 MatrixXd T_dense = MatrixXd::Zero(nrows, ncols);
 MatrixXd F_dense = MatrixXd::Zero(nrows, nrows);


 for(int i=0; i<nrows; i++){
    for(int k=0; k<ncols; k++){
        if(A_dense(i,k) != 0){
            for(int j=0; j<ncols; j++){
                T_dense(i,k) += A_dense(i,j)*S_dense(j,k);               
            }
        }
    }
    for(int k=0; k<ncols; k++){
        if(A_dense(i,k) != 0){
            F_dense(i,i) += T_dense(i,k)*A_dense(i,k); 
        } 
    }     
 }

MatrixXd T_true = A_dense*S_dense;
MatrixXd F_true = T_true*A_dense.transpose();

/*
  std::cout << "T = A*S true: \n" << T_true << std::endl;
  
  std::cout << "T = A*S : \n" << T_dense << std::endl;

  std::cout << "F = A*S*A^T true:\n" << F_true << std::endl;
  std::cout << "F :\n" << F_dense << std::endl;

  std::cout << "norm(diag(F_dense - F_true)) = " << (F_dense.diagonal() - F_true.diagonal()).norm() << std::endl;
*/

#if 0
 // ================================ sparse version ... ==================================== //
 SpMat S = S_dense.sparseView();
 SpMat A = A_dense.sparseView();

 SpMat T(nrows, ncols);
 SpMat F(nrows, nrows);

 T.setZero();
 F.setZero();


 for(int i=0; i<nrows; i++){
    for(int k=0; k<ncols; k++){
        if(A.coeff(i,k) != 0){
            T.insert(i,k) = 0.0;
            for(int j=0; j<ncols; j++){
                T.coeffRef(i,k) += A.coeff(i,j)*S.coeff(j,k);               
            }
        }
    }
    F.insert(i,i) = 0.0;
    for(int k=0; k<ncols; k++){
        F.coeffRef(i,i) += T.coeff(i,k)*A.coeff(i,k);  
    }     
 }

std::cout << "norm(diag(F - F_true)) = " << (F.diagonal() - F_true.diagonal()).norm() << std::endl;
#endif

 // ================================ better sparse version ... ==================================== //


 SpMat S = S_dense.sparseView();
 SpRmMat A = A_dense.sparseView();

 // max number of nonzeros per row are 3 + nb
 // just store 1 column, overwrite every time
 Vect F_vect(nrows);
 F_vect.setZero();

 Vect projMargVar(no);
 get_projMargVar(ns, nt, nb, nnz_invBlks, invBlks.data(), A, projMargVar);

/*
 for(int i=0; i<nrows; i++){
    Eigen::SparseVector<double> T_vect(ncols);
    T_vect.setZero();
    for(int k=0; k<ncols; k++){
        if(A.coeff(i,k) != 0){
            T_vect.insert(k) = 0.0;
            for(int j=0; j<ncols; j++){
                T_vect.coeffRef(k) += A.coeff(i,j)*S.coeff(j,k);               
            }
        }
    }
    for(int k=0; k<ncols; k++){
        if(A.coeff(i,k) != 0){
            F_vect(i) += T_vect.coeff(k)*A.coeff(i,k);  
        }
    }     
 }
 */


// A*S*A^T assuming that A is in row-major, S is ideally in column-major (not so important)
// we are only computing diag(A*S*A^T), thus we only need those entries in S that match sparsity pattern of A
// for an arbitrary sparse/dense matrix S
//std::cout << "S : \n" << S << std::endl;
//std::cout << "A : \n" << A << std::endl;

/*
for (int row = 0; row < nrows; row++){
    printf("\nrow = %d\n", row);
    int arow = row;  
    Eigen::SparseVector<double> T_vect(ncols);
    T_vect.setZero();
    // we only have to iterate through the columns of S, where A(row,col) != 0
    //for(int scol = 0; scol < ncols; scol++){
    for(int sind = A.outerIndexPtr()[row]; sind < A.outerIndexPtr()[row+1]; sind++){
        int scol = A.innerIndexPtr()[sind];    
        // iterating through non-zero entries of each row of A
        T_vect.insert(scol) =0.0;
        for(int aind = A.outerIndexPtr()[row]; aind < A.outerIndexPtr()[row+1]; aind++){   
            int acol = A.innerIndexPtr()[aind]; 
            printf("A: row = %d, col = %d , value = %f\n", row, A.innerIndexPtr()[aind], A.valuePtr()[aind]);
            printf("S(%d, %d) = %f\n", acol, scol, S.coeff(acol, scol));
            // entry of S: S
            // to compute T(row, col) += A(row, acol)*S(acol, col)
            T_vect.coeffRef(scol) += A.valuePtr()[aind]*S.coeff(acol, scol);
        }
    }
    std::cout << "T.row(" << row << ") : " << T_vect.transpose() << std::endl;  
    // now multiply by A^T, 1 scalarproduct in each row, iterate again for outerindexPtr of A  
    for(int aind = A.outerIndexPtr()[row]; aind < A.outerIndexPtr()[row+1]; aind++){   
        int acol = A.innerIndexPtr()[aind]; 
        printf("A: row = %d, col = %d , value = %f\n", row, acol, A.valuePtr()[aind]);
        printf("T(%d) = %f\n", acol, T_vect.coeffRef(acol));
        //printf("A.valuePtr()[aind]*T_vect.coeffRef(acol) = %f\n", A.valuePtr()[aind]*T_vect.coeffRef(acol));
        F_vect(row) += A.valuePtr()[aind]*T_vect.coeffRef(acol);
        printf("F(%d) = %f\n", row, F_vect(row));
    }

}   
*/


  

//std::cout << "T_true : " << MatrixXd(A*S) << std::endl;


//std::cout << "\ndiag(F)" << F_vect.transpose() << std::endl;
//std::cout << "F_true : " << F_true.diagonal().transpose() << std::endl;

//std::cout << "norm(diag(F) - diag(F_true)) = " << (F_vect - F_true.diagonal()). norm() << std::endl;

//std::cout << "F : " << F_vect.transpose() << std::endl;
//std::cout << "norm(diag(F - F_true)) = " << (F_vect - F_true.diagonal()).norm() << std::endl;

exit(1);
#endif 


#if 0
    // ============================================================================== //

    // read test matrix
    int length_tv = 6;
    std::string test_file = "test_vector.dat";
    file_exists(test_file);
    Vect test_vector = read_matrix(test_file, length_tv, 1); 
    std::cout << "test vector : " << test_vector.transpose() << std::endl;

    std::string test_ind_file = "test_ind_vector.dat";
    file_exists(test_ind_file);

    Vect test_ind_vector = read_matrix(test_ind_file, length_tv, 1); 
    std::cout << "test ind vector : " << test_ind_vector.transpose() << std::endl;

    //std::cout << "test_vector " << test_vector.isNaN() << std::endl;

    std::cout << "check if vector has NaN entries: " << std::endl;
    for(int i=0; i<length_tv; i++){
        double a = test_vector[i];
        std::cout << "test_vector[" << i << "] : " << test_vector[i] << ", is NaN: " << isnan(test_vector[i]) << std::endl;
    }

    SpMat A_test = MatrixXd::Random(length_tv,7).sparseView();
    std::cout << "A_test : \n" << A_test << std::endl;

    Vect ATtv = A_test.transpose() * test_vector;
    std::cout << "\nA_test * test_vector: \n" << ATtv.transpose() << std::endl; 

    SpMat A_test_new = A_test;
    Vect test_new_vector = test_vector;

    // TODO: make more efficient! 
    // row/column-wise multiplication with sclar ... points.array().colwise() *= scalars.array();
    // set A matrix values to zero according to y_ind vector
    for(int i = 0; i<length_tv; i++){
        if(test_ind_vector(i) == 0){
            A_test_new.row(i) *= 0;  
            test_new_vector(i) = 0;  
        }
    }

    std::cout << "A_test : \n" << A_test << std::endl;
    std::cout << "A_test_new : \n" << A_test_new << std::endl;

    std::cout << "y_test_new : " << test_new_vector.transpose() << std::endl;

    SpMat ATA_test = A_test_new.transpose() * A_test_new;
    Vect ATy_test  = A_test_new.transpose() * test_new_vector;

    std::cout << "ATA test : \n" << ATA_test << std::endl;
    std::cout << "ATy_test : " << ATy_test.transpose() << std::endl;

    exit(1);
#endif

    // ============================================================================== //


    /* ---------------- read in matrices ---------------- */

#if 1
    // dimension hyperparamter vector
    int dim_th;
    int dim_spatial_domain;

    // spatial component
    SpMat c0; 
    SpMat g1; 
    SpMat g2;

    // spatial-temporal parts
    SpMat g3;
    SpMat M0;
    SpMat M1;
    SpMat M2;

    // data component / fixed effects
    MatrixXd B;
    SpRmMat Ax;
    SpRmMat Ax_all;
    Vect y;

    size_t no_all;

#ifdef PREDICT
    Vect y_all;
    Vect y_ind;
    Vect y_times;
#endif

    bool constr;
    int num_constr;

#ifdef DATA_SYNTHETIC
    constr = false;
#elif defined(DATA_TEMPERATURE)
    //constr = true;
    constr = false;
    num_constr = 1;
#else 
    printf("Invalid dataset.");
    exit(1);
#endif
    
    Vect e;
    MatrixXd Dx;
    MatrixXd Dxy;

    MatrixXd Prec;
    Vect b;
    Vect u;
    double tau;

    if(MPI_rank == 0)
        printf("\n==================== MODEL SPECIFICATIONS ===================\n");

    if(ns == 0 && nt == 0){

        dim_th = 1;

        // #include "generate_regression_data.cpp"
        tau = 0.5;   // is log(precision)

        Dxy.resize(num_constr, nb);
        Dxy << MatrixXd::Ones(num_constr, nb);

        // FOR NOW only SUM-TO-ZERO constraints feasible
        e.resize(num_constr);
        e = Vect::Zero(num_constr);

        if(MPI_rank == 0)
            std::cout << "generate constraint regression data." << std::endl;
        generate_ex_regression_constr(nb, no, tau, Dxy, e, Prec, B, b, y);
        if(MPI_rank == 0)
            std::cout << "b = " << b.transpose() << std::endl;

        // compute true UNCONSTRAINED marginal variances as
        // Q_xy = Q_x + t(B)*Q_e*B, where Q_e = exp(tau)*I  
        MatrixXd Q_xy_true = Prec + exp(tau)*B.transpose()*B;
        MatrixXd Cov_xy_true = Q_xy_true.inverse(); // need to call pardiso here to be efficient ...       
        //std::cout << "Cov_xy_true = \n" << Cov_xy_true << std::endl;
        if(MPI_rank == 0)
            std::cout << "unconstrained variances fixed eff : " << Cov_xy_true.diagonal().transpose() << std::endl;

        // update for constraints  -> would need a solver if Cov_xy_true not available ...
        MatrixXd constraint_Cov_xy_true = Cov_xy_true - Cov_xy_true*Dxy.transpose()*(Dxy*Cov_xy_true*Dxy.transpose()).inverse()*Dxy*Cov_xy_true;
        if(MPI_rank == 0)
            std::cout << "constrained variances fixed eff   : " << constraint_Cov_xy_true.diagonal().transpose() << std::endl;
    
        // read in design matrix 
        // files containing B
        /*std::string B_file        =  base_path + "/B_" + no_s + "_" + nb_s + ".dat";
        file_exists(B_file); 

        // casting no_s as integer
        no = std::stoi(no_s);
        if(MPI_rank == 0){
            std::cout << "total number of observations : " << no << std::endl;
        }

        B = read_matrix(B_file, no, nb);*/

        //std::cout << "y : \n"  << y << std::endl;    
        //std::cout << "B : \n" << B << std::endl;

    } else if(ns > 0 && nt == 1){

        if(MPI_rank == 0)
            std::cout << "spatial model." << std::endl;

        dim_th = 3;

        // check spatial FEM matrices
        std::string c0_file       =  base_path + "/c0_" + ns_s + ".dat";
        file_exists(c0_file);
        std::string g1_file       =  base_path + "/g1_" + ns_s + ".dat";
        file_exists(g1_file);
        std::string g2_file       =  base_path + "/g2_" + ns_s + ".dat";
        file_exists(g2_file);

        // check projection matrix for A.st
        std::string Ax_file     =  base_path + "/Ax_" + to_string(no) + "_" + n_s + ".dat";
        file_exists(Ax_file);

        // read in matrices
        c0 = read_sym_CSC(c0_file);
        g1 = read_sym_CSC(g1_file);
        g2 = read_sym_CSC(g2_file);

        // doesnt require no to be read, can read no from Ax
        Ax = readCSC(Ax_file);
        // get rows from the matrix directly
        // doesnt work for B
        no = Ax.rows();

        // TODO: fix.
        if(constr == true){
            // choose true parameters for theta = log precision observations, 
            Vect theta_original(dim_th);
            theta_original << 0.5, log(4), log(1);  // -> not actually the true parameters!!
            //std::cout << "theta_original = " << theta_original.transpose() << std::endl; 

            // assemble Qs ... 
            MatrixXd Qs = pow(exp(theta_original[1]),2)*(pow(exp(theta_original[2]), 4) * c0 + 2*pow(exp(theta_original[2]),2) * g1 + g2);

            Dx.resize(num_constr, ns);
            Dx << MatrixXd::Ones(num_constr, ns);

            Dxy.resize(num_constr, n);
            Dxy << Dx, MatrixXd::Zero(num_constr, nb);

            // FOR NOW only SUM-TO-ZERO constraints feasible
            e.resize(num_constr);
            e = Vect::Zero(num_constr);            

            if(MPI_rank == 0)
                std::cout << "generate constrained spatial data." << std::endl;
            generate_ex_spatial_constr(ns, nb, no, theta_original, Qs, Ax, Dx, e, Prec, B, b, u, y);


            exit(1);

        }

        /*std::cout << "g1 : \n" << g1.block(0,0,10,10) << std::endl;
        std::cout << "g2 : \n" << g2.block(0,0,10,10) << std::endl;

        std::cout << "Ax : \n" << Ax.block(0,0,10,10) << std::endl;*/

    } else if(ns > 0 && nt > 1) {

        if(MPI_rank == 0)
            std::cout << "spatial-temporal model." << std::endl;
        
        dim_th = 4;

        // files to construct Q.u depending on HYPERPARAMETERS theta
        std::string c0_file      =  base_path + "/c0_" + ns_s + ".dat";
        file_exists(c0_file);
        std::string g1_file      =  base_path + "/g1_" + ns_s + ".dat";
        file_exists(g1_file);
        std::string g2_file      =  base_path + "/g2_" + ns_s + ".dat";
        file_exists(g2_file);
        std::string g3_file      =  base_path + "/g3_" + ns_s + ".dat";
        file_exists(g3_file);

        std::string M0_file      =  base_path + "/M0_" + nt_s + ".dat";
        file_exists(M0_file);
        std::string M1_file      =  base_path + "/M1_" + nt_s + ".dat";
        file_exists(M1_file);
        std::string M2_file      =  base_path + "/M2_" + nt_s + ".dat";
        file_exists(M2_file);  
	
        // read in matrices
        c0 = read_sym_CSC(c0_file);
        g1 = read_sym_CSC(g1_file);
        g2 = read_sym_CSC(g2_file);
        g3 = read_sym_CSC(g3_file);

        M0 = read_sym_CSC(M0_file);
        //arma::mat(M0).submat(0,0,nt-1,nt-1).print();
        M1 = read_sym_CSC(M1_file);
        //arma::mat(M1).submat(0,0,nt-1,nt-1).print();
        M2 = read_sym_CSC(M2_file);
        //arma::mat(M2).submat(0,0,nt-1,nt-1).print();

#ifdef PREDICT

        // check projection matrix for A.st
        // size Ax : no_per_ts*(nt_fit+nt_predict) x (ns*(nt_fit+nt_predict) + nb)
        std::string Ax_file     =  base_path + "/Ax_wNA_" + to_string(no) + "_" + to_string(n) + ".dat";
        file_exists(Ax_file); 

        // keep complete matrix as Ax_all in column major
        Ax = readCSC(Ax_file);

        // cast Ax as row major for fast iteration
        double t_am = - omp_get_wtime();
        //SpMat Ax_all = Ax;
        Ax_all = Ax;
        t_am += omp_get_wtime();

        if(MPI_rank == 0){
            std::cout << "time assign matrix                      : " << t_am << std::endl;
        }

        // get rows from the matrix directly
        // doesnt work for B

        // data y
        size_t no_all = nt_total*no_per_ts;
        std::string y_file        =  base_path + "/y_wNA_" + to_string(no_all) + "_1" + ".dat";
        file_exists(y_file);
        // at this point no is set ... 
        // not a pretty solution. 
        y_all = read_matrix(y_file, no_all, 1);  

        std::string y_ind_file        =  base_path + "/y_indicator_" + to_string(no_all) + "_1" + ".dat";
        file_exists(y_ind_file);
        // at this point no is set ... 
        // not a pretty solution. 
        y_ind = read_matrix(y_ind_file, no_all, 1); 

        std::string y_times_file        =  base_path + "/y_times_" + to_string(no_all) + "_1" + ".dat";
        file_exists(y_times_file);
        // at this point no is set ... 
        // not a pretty solution. 
        y_times = read_matrix(y_times_file, no_all, 1); 

        no = y_ind.sum(); 

        if(MPI_rank == 0){
            std::cout << "total length y : " << no_all << ", total missing : " << no_all - no << std::endl;
            //std::cout << "total number of missing observations: " << no_all - no << std::endl;
            std::cout << "sum(y_ind) = " << y_ind.sum() << std::endl;
            std::cout << "y(1:10) = " << y_all.head(10).transpose() << std::endl;
        }

#else
        // check projection matrix for A.st
        std::string Ax_file     =  base_path + "/Ax_" + no_s + "_" + n_s + ".dat";
        file_exists(Ax_file);

        Ax = readCSC(Ax_file);
        no = Ax.rows();

    // data y
        std::string y_file        =  base_path + "/y_" + no_s + "_1" + ".dat";
        file_exists(y_file);
        // at this point no is set ... 
        // not a pretty solution. 
        y = read_matrix(y_file, no, 1);  
        if(MPI_rank == 0){ 
            std::cout << "sum(y) = " << y.sum() << std::endl;
        }

#endif

        if(MPI_rank == 0){
            //std::cout << "total number of observations : " << no << std::endl;
            std::cout << "read in all matrices." << std::endl;
	    }

    } else {
        if(MPI_rank == 0){
            std::cout << "invalid parameters : ns nt !!" << std::endl;
            exit(1);
        }    
    }

//exit(1);

#endif


/*
#ifdef DATA_SYNTHETIC
    if(constr == false){
        // data y
        std::string y_file        =  base_path + "/y_" + no_s + "_1" + ".dat";
        file_exists(y_file);
        // at this point no is set ... 
        // not a pretty solution. 
        y = read_matrix(y_file, no, 1);
    }
#endif
*/

#ifdef PRINT_MSG
    std::cout << "dim(c0) = " << c0.rows() << " " << c0.cols() << std::endl;
    std::cout << "dim(g1) = " << g1.rows() << " " << g1.cols() << std::endl;
    std::cout << "dim(g2) = " << g2.rows() << " " << g2.cols() << std::endl;
    std::cout << "dim(g3) = " << g3.rows() << " " << g3.cols() << std::endl;
    std::cout << "dim(M0) = " << M0.rows() << " " << M0.cols() << std::endl;
    std::cout << "dim(M1) = " << M1.rows() << " " << M1.cols() << std::endl;
    std::cout << "dim(M2) = " << M2.rows() << " " << M2.cols() << std::endl;
    std::cout << "dim(Ax) = " << Ax.rows() << " " << Ax.cols()<< std::endl;
    std::cout << "dim(y) = " << y_all.size() << std::endl;
#endif


    /* ----------------------- initialise random theta -------------------------------- */

    Vect theta(dim_th);             // define initial guess in model parametrization
    Vect theta_param(dim_th);       // or in interpretable parametrization
    Vect theta_prior_param(dim_th);
    Vect theta_original(dim_th); theta_original.setZero();
    Vect theta_original_param(dim_th); theta_original_param.setZero();

    std::string data_type;

    // initialise theta
    if(ns == 0 && nt == 0){

        theta_original[0] = tau;

        // Initial guess
        theta[0] = 3;
        theta_prior_param[0] = theta[0];


        if(MPI_rank == 0){
            std::cout << "initial theta : "  << theta.transpose() << std::endl; 
        }   

    } else if(ns > 0 && nt == 1){
        //theta << 1, -1, 1;
        //theta << 1, -2, 2;
        //theta_prior << 0, 0, 0;
        //std::cout << "using Elias TOY DATASET" << std::endl;
        // from INLA : log prec Gauss obs, log(Range) for i, log(Stdev) for i     
        //theta_prior << 1.0087220,  -1.0536157, 0.6320466;
        theta_prior_param << 1, -2.3, 2.1;
        theta << theta_prior_param;

        std::cout << "initial theta : "  << theta.transpose() << std::endl;   

    } else {

#ifdef DATA_SYNTHETIC
        data_type = "synthetic";

        // =========== synthetic data set =============== //
        if(MPI_rank == 0){ 
            std::cout << "using SYNTHETIC DATASET" << std::endl; 
        }     
        // constant in conversion between parametrisations changes dep. on spatial dim
        // assuming sphere -> assuming R^3
        dim_spatial_domain = 3;

        // sigma.e (noise observations), sigma.u, range s, range t
        //theta_original_param << 0.5, 4, 1, 10;
        // sigma.e (noise observations), gamma_E, gamma_s, gamma_t
        theta_original << 1.386294, -5.882541,  1.039721,  3.688879;  // here exact solution, here sigma.u = 4
        //theta_prior << 1.386294, -5.594859,  1.039721,  3.688879; // here sigma.u = 3
        // using PC prior, choose lambda  
        theta_prior_param << 0.7/3.0, 0.2*0.7*0.7, 0.7, 0.7/3.0;

        //theta_param << 1.373900, 2.401475, 0.046548, 1.423546; 
        theta_param << 4, 0, 0, 0;
        //theta_param << 4,4,4,4;
        //theta_param << 1.366087, 2.350673, 0.030923, 1.405511;
        
        /*
        if(MPI_rank == 0){
            std::cout << "initial theta      : "  << std::right << std::fixed << theta.transpose() << std::endl;
        }*/


        if(constr == true){

            if(MPI_rank == 0)
                std::cout << "assuming sum-to-zero constraints on spatial-temporal field." << std::endl;
            // sum to zero constraint for latent parameters
            // construct vector (in GMRF book called A, I will call it D) as D=diag(kron(M0, c0)) 
            // in equidistant mesh this would be a vector of all ones, we want sum D_i x_i = 0

            // =============== 1 SUM-TO-ZERO CONSTRAINT PER K TIME-STEPS ==================== //
            // number of time-steps per constraint 
            int tsPerConstr = 100;
            num_constr = ceil(1.0 * nt / tsPerConstr);
            if(MPI_rank == 0)
                std::cout << "num constr = " << num_constr << std::endl;

            if(num_constr*tsPerConstr < nt || tsPerConstr > nt){
                if(MPI_rank == 0)
                    std::cout << "Error! number of constraints * tsPerConstraint not matching nt!! " << num_constr << " " << tsPerConstr << std::endl;
                exit(1);
            }

            // initialize with zero
            Dx.resize(num_constr, ns*nt);
            Dx.setZero();

            Vect M0_diag = M0.diagonal();

            for(int i = 0; i<num_constr; i++){
                int i_start = tsPerConstr*i;
                //std::cout << "i_start = " << i_start << std::endl;
                int nt_int = nt;
                int i_end   = std::min(tsPerConstr*(i+1), nt_int);
                //std::cout << "i_end = " << i_end << std::endl;
                int num_elem = i_end - i_start;

                SpMat M(num_elem, num_elem);
                Vect M_diag = M0_diag.segment(i_start,num_elem);
                M = M0_diag.asDiagonal();
                SpMat D = KroneckerProductSparse<SpMat, SpMat>(M, c0);
                Vect D_diag = D.diagonal();

                //std::cout << "ns*i_end = " << ns*i_end << std::endl;

                for(int j=ns*i_start; j<ns*i_end; j++){
                    //Dx(i,j) = 1.0;
                    Dx(i,j) = D_diag(j);
                }

                //Dx = D.diagonal().transpose();
                //Dx.row(i).segment(ns*i_start, ns*num_elem) << MatrixXd::Ones(1, num_elem);
            }

            /*if(MPI_rank == 0)
                std::cout << Dx << std::endl;*/

            // rescale Dx such that each row sums to one
            for(int i=0; i<num_constr; i++){
                double sum_row = Dx.row(i).sum();
                Dx.row(i) = 1/sum_row*Dx.row(i);
            }

            Dxy.resize(num_constr, n);
            Dxy << Dx, MatrixXd::Zero(num_constr, nb);

            // FOR NOW only SUM-TO-ZERO constraints feasible
            e.resize(num_constr);
            e = Vect::Zero(num_constr);     

            //exit(1);       

#if 0
            if(MPI_rank == 0)
                std::cout << "constrain spatial-temporal data." << std::endl;
            
            // assemble Qst ... 
            SpMat Qst(ns*nt, ns*nt);
            construct_Q_spat_temp(theta_original, c0, g1, g2, g3, M0, M1, M2, Qst);
            MatrixXd Qst_d = MatrixXd(Qst);

            generate_ex_spatial_temporal_constr(ns, nt, nb, no, theta_original, Qst_d, Ax, Dx, e, Prec, b, u, y);

            Vect x(n);
            x << u, b;
            //std::cout << "true x = " << x.transpose() << std::endl;

            // compute true UNCONSTRAINED marginal variances as
            MatrixXd Q_x(n,n);
            Q_x << Qst_d, MatrixXd::Zero(ns*nt, nb), MatrixXd::Zero(nb, ns*nt), Prec;
            //std::cout << "Q_x : \n" << Q_x.block(0,0,20,20) << std::endl;
            //std::cout << "dim(Q_x) = " << Q_x.rows() << " " << Q_x.cols() << std::endl;
            //std::cout << "Q_x : \n" << Q_x(127,127) << std::endl; //Q_x.block(110,110,127,127)

            //Q_xy = Q_x + t(Ax)*Q_e*Ax, where Q_e = exp(tau)*I  -> how to incorporate constraints?
            MatrixXd Q_xy_true = Q_x + exp(theta_original[0])*Ax.transpose()*Ax;
            MatrixXd Cov_xy_true = Q_xy_true.inverse(); // need to call pardiso here to be efficient ...       
            //std::cout << "Cov_xy_true = \n" << Cov_xy_true << std::endl;
            if(MPI_rank == 0)
                std::cout << "unconstr. sd. fixed eff : " << Cov_xy_true.diagonal().tail(nb).cwiseSqrt().transpose() << std::endl;

            // update for constraints
            MatrixXd constraint_Cov_xy_true = Cov_xy_true - Cov_xy_true*Dxy.transpose()*(Dxy*Cov_xy_true*Dxy.transpose()).inverse()*Dxy*Cov_xy_true;
            if(MPI_rank == 0)
                std::cout << "constr. sd. fixed eff   : " << constraint_Cov_xy_true.diagonal().tail(nb).cwiseSqrt().transpose() << std::endl;

            //exit(1);
#endif            


        }

#elif defined(DATA_TEMPERATURE)

        // =========== temperature data set =============== //
        data_type = "temperature";

               // constant in conversion between parametrisations changes dep. on spatial dim
        // assuming sphere -> assuming R^3
        dim_spatial_domain = 3;

        if(MPI_rank == 0){
            std::cout << "using TEMPERATURE DATASET assuming dim spatial domain : " << dim_spatial_domain << std::endl; 
            if(constr)
                std::cout << "assuming sum-to-zero constraints on spatial-temporal field." << std::endl;
        }

        //theta << 4, 4, 4, 4;    //
        //theta_param << 4, 0, 0, 0;
        //theta_param << -1.308664,  0.498426,  4.776162,  1.451209;
        //theta_param << -1.5, 7, 7, 3;
        //theta_param << 4, 1, 1, 0;
        theta_param << -2, -0.5, 5, 2;

        //theta_original << -1.269613,  5.424197, -8.734293, -6.026165;  // estimated from INLA / same for my code varies a bit according to problem size
	//theta_original_param << -2.090, 9.245, 11.976, 2.997; // estimated from INLA

        // using PC prior, choose lambda  
	// previous order : interpret_theta & lambda order : sigma.e, range t, range s, sigma.u -> doesn't match anymore
	// NEW ORDER sigma.e, range s, range t, sigma.u 
        //theta_prior_param << 0.7/3.0, 0.2*0.7*0.7, 0.7, 0.7/3.0;
	// NEW ORDER sigma.e, range s, range t, sigma.u
	// -log(p)/u where c(u, p)
	theta_prior_param[0] = -log(0.01)/5; 	      //prior.sigma obs : 5, 0.01
	//theta_prior_param[1] = -log(0.5)/1000;        //prior.rs=c(1000, 0.5), ## P(range_s < 1000) = 0.5
	theta_prior_param[1] = -log(0.01)/500;        
    //theta_prior_param[2] = -log(0.5)/20;	      //prior.rt=c(20, 0.5), ## P(range_t < 20) = 0.5
	theta_prior_param[2] = -log(0.01)/1;
    //theta_prior_param[3] = -log(0.5)/10;          //prior.sigma=c(10, 0.5) ## P(sigma_u > 10) = 0.5
    theta_prior_param[3] = -log(0.01)/3;	    

	
        //std::cout << "theta prior        : " << std::right << std::fixed << theta_prior.transpose() << std::endl;
        //theta << -0.2, -2, -2, 3;
        /*if(MPI_rank == 0){
            std::cout << "initial theta      : "  << std::right << std::fixed << theta.transpose() << std::endl;
        }*/

        if(constr){

#if 1
            // =============== 1 SUM-TO-ZERO CONSTRAINT PER K TIME-STEPS ==================== //
            // number of time-steps per constraint 
            int tsPerConstr = nt;
            num_constr = ceil(1.0 * nt / tsPerConstr);
            if(MPI_rank == 0)
                std::cout << "num constr = " << num_constr << std::endl;

            if(num_constr*tsPerConstr < nt || tsPerConstr > nt){
                if(MPI_rank == 0)
                    std::cout << "Error! number of constraints * tsPerConstraint not matching nt!! " << num_constr << " " << tsPerConstr << std::endl;
                exit(1);
            }

            // initialize with zero
            Dx.resize(num_constr, ns*nt);
            Dx.setZero();

            Vect M0_diag = M0.diagonal();

            for(int i = 0; i<num_constr; i++){
                int i_start = tsPerConstr*i;
                //std::cout << "i_start = " << i_start << std::endl;
                int nt_int = nt;
                int i_end   = std::min(tsPerConstr*(i+1), nt_int);
                //std::cout << "i_end = " << i_end << std::endl;
                int num_elem = i_end - i_start;

                SpMat M(num_elem, num_elem);
                Vect M_diag = M0_diag.segment(i_start,num_elem);
                M = M0_diag.asDiagonal();
                SpMat D = KroneckerProductSparse<SpMat, SpMat>(M, c0);
                Vect D_diag = D.diagonal();

                //std::cout << "ns*i_end = " << ns*i_end << std::endl;

                for(int j=ns*i_start; j<ns*i_end; j++){
                    //Dx(i,j) = 1.0;
                    Dx(i,j) = D_diag(j);
                }

                //Dx = D.diagonal().transpose();
                //Dx.row(i).segment(ns*i_start, ns*num_elem) << MatrixXd::Ones(1, num_elem);
            }

            /*if(MPI_rank == 0)
                std::cout << Dx << std::endl;*/

            // rescale Dx such that each row sums to one
            for(int i=0; i<num_constr; i++){
                double sum_row = Dx.row(i).sum();
                Dx.row(i) = 1/sum_row*Dx.row(i);
            }

            Dxy.resize(num_constr, n);
            Dxy << Dx, MatrixXd::Zero(num_constr, nb);

            // FOR NOW only SUM-TO-ZERO constraints feasible
            e.resize(num_constr);
            e = Vect::Zero(num_constr);  
            
#endif

            // set up constraints Dx = e
            /*Dx.resize(num_constr, ns*nt);
            SpMat D = KroneckerProductSparse<SpMat, SpMat>(M0, c0);
            Dx = D.diagonal().transpose();
            if(MPI_rank == 0){
                std::cout << "sum(Dx)  = " << Dx.row(0).sum() << std::endl;
                //std::cout << "Dx(1:50) = " << Dx.block(0,0,1,50) << std::endl;
            }
            //Dx << MatrixXd::Ones(num_constr, ns*nt);

            // rescale Dx such that each row sums to one
            for(int i=0; i<num_constr; i++){
                double sum_row = Dx.row(i).sum();
                Dx = 1/sum_row*Dx;
            }

            if(MPI_rank == 0){
                std::cout << "sum(Dx)  = " << Dx.row(0).sum() << std::endl;
                //std::cout << "Dx(1:50) = " << Dx.block(0,0,1,50) << std::endl;
            }

            Dxy.resize(num_constr, n);
            Dxy << Dx, MatrixXd::Zero(num_constr, nb);


            // FOR NOW only SUM-TO-ZERO constraints possible
            e.resize(num_constr);
            e = Vect::Zero(num_constr); 

            //exit(1);*/
        }

#else 
        std::cerr << "\nUnknown datatype! Choose synthetic or temperature dataset!" << std::endl;
        exit(1);
#endif

    }

    // ========================== set up validation set ======================= //

    bool validate = false;
    //bool validate = true;
    Vect w;

    if(validate){
        // Vect::Random() creates uniformly distributed random vector between [-1,1]
        // size validation set, ie. 0.1 -> 10% of total observations
        double r = 0.1;
        w = Vect::Random(no);

        for(int i=0;i<no; i++){
            if(w[i] < 1 - 2*r){
                w[i] = 1;
            } else {
                w[i] = 0;
            }
        }

        if(MPI_rank == 0)
            std::cout << "size validation set = " << no - w.sum() << std::endl;
        //std::cout << "w = " << w.transpose() << std::endl;
    }

    //exit(1);
 
    // ============================ set up BFGS solver ======================== //

    // Set up parameters
    LBFGSParam<double> param;    
    // set convergence criteria
    // stop if norm of gradient smaller than :
    // computed as ||𝑔|| < 𝜖 ⋅ max(1,||𝑥||)
    param.epsilon = 1e-1;
    // param.epsilon = 1e-2; // ref sol
    // or if objective function has not decreased by more than  
    // cant find epsilon_rel in documentation ...
    // stops if grad.norm() < eps_rel*x.norm() 
    param.epsilon_rel=1e-3;
    //param.epsilon_rel=1e-4; // ref sol
    //param.epsilon_rel = 1e-5;
    // in the past ... steps
    param.past = 2;
    // TODO: stepsize too small? seems like it almost always accepts step first step.    
    // changed BFGS convergence criterion, now stopping when abs(f(x_k) - f(x_k-1)) < delta
    // is this sufficiently bullet proof?!
    //param.delta = 1e-3;
    param.delta = 1e-3;
    //param.delta = 1e-9; // ref sol
    // maximum line search iterations
    param.max_iterations = 200; //200;

    // Create solver and function object
    LBFGSSolver<double> solver(param);

    /*std::cout << "\nspatial grid size  : " << std::right << std::fixed << g1.rows() << " " << g1.cols() << std::endl;
    std::cout << "temporal grid size : " << M1.rows() << " " << M1.cols() << std::endl;
    std::cout << "Ax size            : " << Ax.rows() << " " << Ax.cols() << std::endl;*/

    // ============================ set up Loop over each time step ======================== //

    // int t_last = 3;
    // for(int t_init=0; t<t_last; t_init++){
    //
    // 
    int nt_init_fit  = 0;
    int nt_last_fit  = nt_init_fit + nt_fit - 1;
    int nt_init_pred = nt_last_fit + 1; 
    int nt_last_pred = nt_init_pred + nt_pred - 1;

    if(MPI_rank == 0){
        printf("nt_init_fit: %d, nt_last_fit: %d, nt_init_pred: %d, nt_last_pred: %d\n", nt_init_fit, nt_last_fit, nt_init_pred, nt_last_pred);
    }

    if(nt_last_pred > nt_total){
        printf("nt_last_pred (%d) exceeds nt_total (%ld)!\n", nt_last_pred, nt_total);
        exit(1);
    }

    // 

    // extract appropriate columns from y
    y = y_all(Eigen::seq(nt_init_fit*no_per_ts, (nt_last_pred+1)*no_per_ts-1));
    Vect y_ind_sub = y_ind(Eigen::seq(nt_init_fit*no_per_ts, (nt_last_pred+1)*no_per_ts-1));

    no = y_ind_sub.sum();
    if(MPI_rank == 0){
        printf("first index: %ld, last index: %ld\n", nt_init_fit*no_per_ts, (nt_last_pred+1)*no_per_ts-1);
        printf("length(y) = %ld, no(w/out NA) = %ld, rows(Ax) = %ld\n", y.size(), no, Ax.rows());
    }

    // set A matrix values to zero according to y_ind vector
    // iterating through rows, make sure Ax is row major!
    double t_mm = - omp_get_wtime();
    for(int i = 0; i<Ax.rows(); i++){
        if(y_ind_sub(i) == 0){
            Ax.row(i) *= 0;  
            y(i) = 0.0;  
        }
    }
    t_mm += omp_get_wtime();
    
    if(MPI_rank == 0){
        std::cout << "time matrix row multiply                : " << t_mm << std::endl;
    }

    // ============================ set up Posterior of theta ======================== //

    //std::optional<PostTheta> fun;
    PostTheta* fun;

    if(ns == 0){
        // fun.emplace(nb, no, B, y);
        if(MPI_rank == 0){
            std::cout << "Call constructor for regression model." << std::endl;
        }
        fun = new PostTheta(ns, nt, nb, no, B, y, theta_prior_param, solver_type, constr, Dxy, validate, w);
    } else if(ns > 0 && nt == 1) {
        if(MPI_rank == 0){
            std::cout << "\ncall spatial constructor." << std::endl;
        }
        // PostTheta fun(nb, no, B, y);
        fun = new PostTheta(ns, nt, nb, no, Ax, y, c0, g1, g2, theta_prior_param, solver_type, dim_spatial_domain, constr, Dx, Dxy, validate, w);
    } else {
        if(MPI_rank == 0){
            std::cout << "\ncall spatial-temporal constructor." << std::endl;
        }
        fun = new PostTheta(ns, nt, nb, no, Ax, y, c0, g1, g2, g3, M0, M1, M2, theta_prior_param, solver_type, dim_spatial_domain, constr, Dx, Dxy, validate, w);
    }

    if(MPI_rank == 0)
        printf("\n======================= HYPERPARAMETERS =====================\n");

    if(MPI_rank == 0){
        std::cout << "theta original     : " << std::right << std::fixed << theta_original.transpose() << std::endl;
        std::cout << "theta prior param  : " << theta_prior_param.transpose() << std::endl;
    }

    // convert from interpretable parametrisation to internal one
    if(dim_th == 4){
	   theta[0] = theta_param[0];
        fun->convert_interpret2theta(theta_param[1], theta_param[2], theta_param[3], theta[1], theta[2], theta[3]);

        if(MPI_rank == 0){
            Vect theta_interpret_initial(dim_th);
            theta_interpret_initial[0] = theta[0];
            fun->convert_theta2interpret(theta[1], theta[2], theta[3], theta_interpret_initial[1], theta_interpret_initial[2], theta_interpret_initial[3]);
            std::cout << "theta interpret. param. : " << theta_param.transpose() << std::endl;
	    std::cout << "initial theta      : "  << std::right << std::fixed << theta.transpose() << std::endl;
            std::cout << "initial theta interpret. param. : " << theta_interpret_initial.transpose() << std::endl;
        }
    }

    //exit(1);

#ifdef WRITE_RESULTS
   //string results_folder = base_path + "/results_param_fixed_inverse";
    string results_folder = base_path + "/results_param_INLAmode";
   if(MPI_rank == 0){
    	create_folder(results_folder);
   }
#endif

#if 0

if(MPI_rank == 0){
    std::cout << "\n================== Comparison Q matrices. =====================\n" << std::endl;
    //theta_param << -2.21506094147639, 8.01123771552475, 7.007131066306, 3.09174134657989;
    //theta_param << -2.48703929737538, 7.83456608733632, 6.89882091727513, 2.84522770375556;
    //theta_param << -2.485177, 7.860605, 7.066556,  2.555450;
    //theta_param << -2.17010036733188, 9.06814137004373, 10.3337426479875, 3.21366330420807;
    //theta_param << -2.48041519338178, 7.68975975256294, 6.91277775656503, 2.70184818295968;
    theta_param << -2.484481, 7.836006, 7.023295, 2.504872;

    std::cout << "theta param : " << theta_param.transpose() << std::endl;
    theta[0] = theta_param[0];
    fun->convert_interpret2theta(theta_param[1], theta_param[2], theta_param[3], theta[1], theta[2], theta[3]);
    std::cout << "theta       : " << theta.transpose() << std::endl;

    // includes prior fixed effects
    SpMat Qst_INLA(n,n);
    std::string file_name_Qst_INLA = base_path + "/final_Qprior_" + to_string(n) + ".dat";
    Qst_INLA = read_sym_CSC(file_name_Qst_INLA);
    std::cout << "read in Qst_INLA. norm(Qst_INLA) = " << Qst_INLA.norm() << std::endl;
    std::cout << "Qst_INLA[1:10,1:10] : \n" << Qst_INLA.block(0,0,9,9) << std::endl;

    // doesnt include fixed effects
    SpMat Qst(n-nb,n-nb);
    fun->construct_Q_spat_temp(theta, Qst);
    std::cout << "Qst[1:10,1:10] = \n" << Qst.block(0,0,9,9) << std::endl;
    std::cout << "constucted Qst. norm(Qst) : " << Qst.norm() << std::endl;

    SpMat Q_INLA(n,n);
    std::string file_name_Q_INLA = base_path + "/final_Q_" + to_string(n) + ".dat";
    Q_INLA = read_sym_CSC(file_name_Q_INLA);
    MatrixXd Q_INLA_dense = MatrixXd(Q_INLA);
    std::cout << "read in Q_INLA. norm(Q_INLA) = " << Q_INLA.norm() << std::endl;
    std::cout << "Q_INLA(bottomRightCorner) : \n" << Q_INLA_dense.bottomRightCorner(10,10) << std::endl;

    SpMat Q(n,n);
    //theta <<
    fun->construct_Q(theta, Q);
    MatrixXd Q_dense = MatrixXd(Q);
    std::cout << "Q[1:10,1:10] = \n" << Q.block(0,0,9,9) << std::endl;
    std::cout << "constucted Q. norm(Q) = " << Q.norm() << std::endl;
    std::cout << "Q(bottomRightCorner) : \n" << Q_dense.bottomRightCorner(10,10) << std::endl;


    //std::cout << "norm(Q-Q_INLA) = " << (Q-Q_INLA).norm() << std::endl;
}

#endif

    double fx;

#if 0
	
    //theta_param << -2.152, 9.534, 11.927, 3.245;
   
    theta[0] = theta_param[0];
    fun->convert_interpret2theta(theta_param[1], theta_param[2], theta_param[3], theta[1], theta[2], theta[3]);

    if(MPI_rank == 0){
        std::cout << "theta param : " << theta_param.transpose() << std::endl;
        std::cout << "theta       : " << theta.transpose() << std::endl;
    }

    double t_f_eval = -omp_get_wtime();

    ArrayXi fact_to_rank_list(2);
    fact_to_rank_list << 0,0;
    /*if(MPI_size >= 2){
        fact_to_rank_list[1] = 1; 
        }
    std::cout << "i = " << 0 << ", MPI rank = " << MPI_rank << ", fact_to_rank_list = " << fact_to_rank_list.transpose() << std::endl;
      */      
    if(MPI_rank == fact_to_rank_list[0] || MPI_rank == fact_to_rank_list[1]){

        // single function evaluation
        for(int i=0; i<5; i++){

            Vect mu_dummy(n);
        double t_temp = -omp_get_wtime();
            fx = fun->eval_post_theta(theta, mu_dummy);
            //fx = fun->eval_post_theta(theta_original, mu_dummy);
        t_temp += omp_get_wtime();

                if(MPI_rank == fact_to_rank_list[0])
            std::cout <<  "f(x) = " << fx << ", time : " << t_temp << " sec. " << std::endl;

        }
    }

    t_f_eval += omp_get_wtime();
    if(MPI_rank == fact_to_rank_list[0])
        std::cout << "time in f eval loop : " << t_f_eval << std::endl;

#endif 

double time_bfgs = 0.0;

#if 0
    if(MPI_rank == 0)
        printf("\n====================== CALL BFGS SOLVER =====================\n");

    //LIKWID_MARKER_INIT;
    //LIKWID_MARKER_THREADINIT;

    //theta_param << -1.5, 7, 7, 3;
    //theta_param << -2.484481  7.836006  7.023295  2.504872
    //theta_param << -1.5, 8, 8, 3;

    //theta_param << -2.15, 9.57, 11.83, 3.24;

    theta[0] = theta_param[0];
    fun->convert_interpret2theta(theta_param[1], theta_param[2], theta_param[3], theta[1], theta[2], theta[3]);

    if(MPI_rank == 0){    
        std::cout << "theta param : " << theta_param.transpose() << std::endl;
        std::cout << "theta       : " << theta.transpose() << std::endl;
    }

    time_bfgs = -omp_get_wtime();
    int niter = solver.minimize(*fun, theta, fx, MPI_rank);

    //LIKWID_MARKER_CLOSE;

    time_bfgs += omp_get_wtime();

    // get number of function evaluations.
    int fn_calls = fun->get_fct_count();

    if(MPI_rank == 0){
        std::cout << niter << " iterations and " << fn_calls << " fn calls." << std::endl;
        //std::cout << "time BFGS solver             : " << time_bfgs << " sec" << std::endl;

        std::cout << "\nf(x)                         : " << fx << std::endl;
    }

    /*int fct_count = fun->get_fct_count();
    std::cout << "function counts thread zero  : " << fct_count << std::endl;*/

    Vect grad = fun->get_grad();
    if(MPI_rank == 0){
        std::cout << "grad                         : " << grad.transpose() << std::endl;
    }


#ifdef WRITE_RESULTS
    if(MPI_rank == 0){
    	std::string file_name_theta = results_folder + "/mode_theta_interpret_param.txt";
    	theta_param[0] = theta[0];
    	fun->convert_theta2interpret(theta[1], theta[2], theta[3], theta_param[1], theta_param[2], theta_param[3]);
    	write_vector(file_name_theta, theta_param, dim_th);
    }
#endif

    /*std::cout << "\nestimated mean theta         : " << theta.transpose() << std::endl;
    std::cout << "original theta               : " << theta_prior.transpose() << "\n" << std::endl;*/

    /*double eps = 0.005;
    Vect temp(4);
    temp << -5,2,3,-2;
    double f_temp = fun->f_eval(temp);
    std::cout << "f eval test : " << f_temp << endl;
    MatrixXd cov = fun->get_Covariance(temp, eps);
    std::cout << "estimated covariance theta with epsilon = " << eps << "  :  \n" << cov << std::endl;*/

    if(MPI_rank == 0){
        std::cout << "\norig. mean parameters        : " << theta_original.transpose() << std::endl;
        std::cout << "est.  mean parameters        : " << theta.transpose() << std::endl;
    }

    // convert between different theta parametrisations
    if(dim_th == 4 && MPI_rank == 0){
        theta_original_param[0] = theta_original[0];
        fun->convert_theta2interpret(theta_original[1], theta_original[2], theta_original[3], theta_original_param[1], theta_original_param[2], theta_original_param[3]);
        //std::cout << "\norig. mean interpret. param. : " << theta_original[0] << " " << prior_ranT << " " << prior_ranS << " " << prior_sigU << std::endl;
        std::cout << "\norig. mean interpret. param. : " << theta_original_param[0] << " " << theta_original_param[1] << " " << theta_original_param[2] << " " << theta_original_param[3] << std::endl;

        double lgamE = theta[1]; double lgamS = theta[2]; double lgamT = theta[3];
        double sigU; double ranS; double ranT;
        fun->convert_theta2interpret(lgamE, lgamS, lgamT, ranS, ranT, sigU);
        std::cout << "est.  mean interpret. param. : " << theta[0] << " " << ranS << " " << ranT << " " << sigU << std::endl;
    }

#endif // end BFGS optimize

 double t_get_covariance = 0.0;

#if 0
    Vect theta_max(dim_th);
    //theta_max << -2.15, 9.57, 11.83, 3.24;    // theta
    //theta_max = theta_prior;
    theta_max = theta;

    // in what parametrisation are INLA's results ... ?? 
    double eps = 0.005;
    MatrixXd cov(dim_th,dim_th);

#if 0
    double t_get_covariance = -omp_get_wtime();

    eps = 0.005;
    //cov = fun->get_Covariance(theta_max, sqrt(eps));
    cov = fun->get_Covariance(theta_max, eps);

    t_get_covariance += omp_get_wtime();

    if(MPI_rank ==0){
        std::cout << "covariance                   : \n" << cov << std::endl;
        std::cout << "time get covariance          : " << t_get_covariance << " sec" << std::endl;
    }
#endif


#if 1
    //convert to interpretable parameters
    // order of variables : gaussian obs, range t, range s, sigma u
    Vect interpret_theta(4);
    interpret_theta[0] = theta_max[0];
    fun->convert_theta2interpret(theta_max[1], theta_max[2], theta_max[3], interpret_theta[1], interpret_theta[2], interpret_theta[3]);
   
    //interpret_theta << -2.152, 9.679, 12.015, 3.382;
#ifdef PRINT_MSG 
    if(MPI_rank == 0){
        std::cout << "est. Hessian at theta param : " << interpret_theta[0] << " " << interpret_theta[1] << " " << interpret_theta[2] << " " << interpret_theta[3] << std::endl;
    }
#endif

    //double t_get_covariance = -omp_get_wtime();
    t_get_covariance = -omp_get_wtime();
    cov = fun->get_Cov_interpret_param(interpret_theta, eps);
    t_get_covariance += omp_get_wtime();


    if(MPI_rank == 0){
        std::cout << "\ncovariance interpr. param.  : \n" << cov << std::endl;
        std::cout << "\nsd hyperparameters interpr. :   " << cov.diagonal().cwiseSqrt().transpose() << std::endl;
        //std::cout << "time get covariance         : " << t_get_covariance << " sec" << std::endl;

#ifdef WRITE_RESULTS
    std::string file_name_cov = results_folder + "/inv_hessian_mode_theta_interpret_param.txt";
    write_matrix(file_name_cov, cov);
#endif
    }

#endif

#endif // end get covariance


#if 1

    /*
    theta_param << -1.407132, 8.847758, 9.986088, 3.783090;
    //theta_param << -1.40687802, 9.34301129, 11.00926400, 4.28259598;
    //theta_param << -1.407039,  8.841431,  9.956879,  3.770581;
    // theta_param << -1.40701328482976, 9.34039748237832, 11.0020161941741, 4.27820007271347;
    theta[0] = theta_param[0];
    fun->convert_interpret2theta(theta_param[1], theta_param[2], theta_param[3], theta[1], theta[2], theta[3]);
    //theta << -1.407039, -7.801710, -6.339689, 5.588888;

    if(MPI_rank == 0){
        std::cout << "Computing mean latent parameters using theta interpret : " << theta_param.transpose() << std::endl;
    }
    */
    theta << -1.99803915, -9.82895738,  1.98118689,  8.288427241;
    
    double t_get_fixed_eff;
    Vect mu(n);

    ArrayXi fact_to_rank_list(2);
    fact_to_rank_list << 0,0;
    /*if(MPI_size >= 2){
        fact_to_rank_list[1] = 1; 
    }*/

    if(MPI_rank == fact_to_rank_list[0] || MPI_rank == fact_to_rank_list[1]){
        //std::cout << "MPI rank = " << MPI_rank << ", fact_to_rank_list = " << fact_to_rank_list.transpose() << std::endl;

        t_get_fixed_eff = - omp_get_wtime();
        //fun->get_mu(theta, mu, fact_to_rank_list);
        fun->get_mu(theta, mu);
	t_get_fixed_eff += omp_get_wtime();
    }

    // CAREFUL! at the moment mu is in rank 1 ... how to do this properly??
    if(MPI_rank == fact_to_rank_list[1]){

        std::cout << "\nestimated mean fixed effects : " << mu.tail(nb).transpose() << std::endl;
        std::cout << "estimated mean random effects: " << mu.head(10).transpose() << std::endl;
        //std::cout << "time get fixed effects       : " << t_get_fixed_eff << " sec\n" << std::endl;

//#ifdef PRINT_MSG
        if(constr == true)
            std::cout << "Dxy*mu : " << (Dxy*mu).transpose() << ", \nshould be : " << e.transpose() << std::endl;
//#endif

#ifdef WRITE_RESULTS
    std::string file_name_fixed_eff = results_folder + "/mean_latent_parameters.txt";
    write_vector(file_name_fixed_eff, mu, n);
#endif

#endif // endif get_mu()

// write matrix to file for debugging ... 
#if 0
        SpMat Qprior(ns*nt,ns*nt);
        fun->construct_Q_spat_temp(theta, Qprior);
        
        std::cout << "Qprior(1:10,1:10):\n" << Qprior.block(0,0,20,20) << std::endl;
        std::string Qprior_file = base_path + "/Qprior_" + n_s + "_" + n_s + ".dat";
        write_sym_CSC_matrix(Qprior_file, Qprior); 
         
        
        // write to file first time step
        SpMat Qprior_1stTs =  Qprior.block(0,0,ns,ns);
        std::cout << "dim(Qprior_1stTs) = " << Qprior_1stTs.rows() << "_" << Qprior_1stTs.cols() << std::endl;
        std::string Qprior_1stTs_file = base_path + "/Qprior_1stTs_" + ns_s + "_" + ns_s + ".dat";
        write_sym_CSC_matrix(Qprior_1stTs_file, Qprior_1stTs);  

        
        SpMat Q(n,n);
        fun->construct_Q(theta, Q);
        std::cout << "Q(1:10,1:10):\n" << Q.block(0,0,20,20) << std::endl;
        std::string Q_file = base_path + "/Qxy_" + n_s + "_" + n_s + ".dat";
        write_sym_CSC_matrix(Q_file, Q);  

        std::cout << "Q(1:10,1:10)-Qprior(1:10,1:10):\n" << Q.block(0,0,20,20) -  Qprior.block(0,0,20,20) << std::endl;

        Vect b(n);
        fun->construct_b(theta, b);
        std::cout << "b(1:10): " << b.head(10).transpose() << std::endl;
        std::string b_file = base_path + "/b_xy_" + n_s + "_1.dat";
        write_vector(b_file, b, n);
    

#endif


#ifdef PREDICT
    // =================================== compute prediction =================================== //

    // make prediction for mean at all locations y using Ax * mu
    Vect y_predict = Ax_all * mu;

#ifdef WRITE_RESULTS
        std::string file_name_y_predict_mean = results_folder + "/y_predict_mean_" + to_string(y_predict.size()) + ".txt";
        // contains prediction for all y, not just previously unknown 
        // if only those needed, filter by indicator vector
        //printf("no all : %ld, rows(Ax_all) : %ld\n", no_all, Ax_all.rows());
        write_vector(file_name_y_predict_mean, y_predict, y_predict.size());
#endif

#endif // end predict


    } // end if(MPI_rank == fact_to_rank_list[1]), get_mu()
  
    // =================================== compute marginal variances =================================== //
#if 1
    double t_get_marginals;
    Vect marg(n);

    // when the range of u is large the variance of b0 is large.
    if(MPI_rank == 0){
        std::cout << "\n==================== compute marginal variances ================" << std::endl;
        //theta << -1.269613,  5.424197, -8.734293, -6.026165; // most common solution for temperature dataset
        std::cout << "\nUSING ESTIMATED THETA : " << theta.transpose() << std::endl;


    	t_get_marginals = -omp_get_wtime();
    	fun->get_marginals_f(theta, marg);
    	t_get_marginals += omp_get_wtime();

        //std::cout << "\nest. variances fixed eff.    :  " << marg.tail(10).transpose() << std::endl;
        std::cout << "est. standard dev fixed eff  : " << marg.tail(nb).cwiseSqrt().transpose() << std::endl;
        std::cout << "est. std dev random eff      : " << marg.head(10).cwiseSqrt().transpose() << std::endl;
        //std::cout << "diag(Cov) :                     " << Cov.diagonal().transpose() << std::endl;

#ifdef WRITE_RESULTS
    	std::string file_name_marg = results_folder + "/sd_latent_parameters.txt";
    	write_vector(file_name_marg, marg.cwiseSqrt(), n);
#endif

#ifdef PREDICT

       // get marginal variances for all locations y using A*inv(Q)*A^T
       // TODOL QinvSp as rowmajor ...
       SpMat QinvSp(n,n);
       fun->get_fullFact_marginals_f(theta, QinvSp);
       
      std::cout << "QinvSp: est. standard dev fixed eff  : " << QinvSp.diagonal().tail(nb).cwiseSqrt().transpose() << std::endl;
      std::cout << "QinvSp: est. std dev random eff      : " << QinvSp.diagonal().head(10).cwiseSqrt().transpose() << std::endl;

       std::cout << "nnz(Ax_all) = " << Ax_all.nonZeros() << ", nnz(QinvSp) = " << QinvSp.nonZeros();
       std::cout << ", dim(QinvSp) = " << QinvSp.rows() << " " << QinvSp.cols() << ", dim(Ax_all) = " << Ax_all.rows() << " " << Ax_all.cols() << std::endl;

       SpRmMat temp = Ax_all;
       SpRmMat temp2 = Ax_all;

       //double* temp_array = new double[Ax_all.nonZeros()];

       double t_firstMult = - omp_get_wtime();

       // write my own multiplication
       // for each row in Ax_all iterate through the columns of QinvSp -> only consider nonzero entries of Ax_all
 
        int counter = 0;
       for (int k=0; k<temp2.outerSize(); ++k){
			for (SparseMatrix<double, RowMajor>::InnerIterator it(temp2,k); it; ++it)
			{
			    // access pattern is row-major -> can i directly write to 
                it.valueRef() = (Ax_all.row(it.row())).dot(QinvSp.col(it.col()));
                //temp.valuePtr()[counter] = (Ax_all.row(it.row())).dot(QinvSp.col(it.col()));
                //temp_array[counter] = (Ax_all.row(it.row())).dot(QinvSp.col(it.col()));
                //temp.valuePtr()[counter] = temp1;
                counter++;
			}
        } // end outer for loop
    
       t_firstMult += omp_get_wtime();
       printf("time 1st Mult innerIter : %f\n", t_firstMult);
       //exit(1);

       //printf("norm(temp - temp2) = %f\n", (temp-temp2).norm());

       Vect projMargVar(Ax_all.rows());
       //Vect projMargVar2(Ax_all.rows());

       double t_secondMult = - omp_get_wtime();
       for(int i=0; i<Ax_all.rows(); i++){
            //projMargVar(i) = (temp.row(i)).dot(Ax_ex.row(i));
            projMargVar(i) = (temp.row(i)).dot(Ax_all.row(i));
       }
       t_secondMult += omp_get_wtime();
       printf("time 2nd Mult: %f\n", t_secondMult);

       //printf("norm(projMargVar - projMargVar2) = %f\n", (projMargVar - projMargVar2).norm());
       //Vect projMargVar = QinvProj.diagonal();

       printf("size(projMargVar) : %ld, no_all : %ld\n", projMargVar.size(), no_all);
       
       //std::cout << "est. std dev 1st 10 loc            : " << projMargVar.head(20).cwiseSqrt().transpose() << std::endl;

       // inv(Q) comes in the form of double* invBlks -> CAREFUL particular order
       //void get_projMargVar(size_t ns, size_t nt, size_t nb, size_t nnz_invBlks, double* invBlks, SpRmMat& A, Vect& projMargVar);
       
       Vect projMargSd(Ax_all.rows());
       projMargSd = projMargVar.cwiseSqrt();

#ifdef WRITE_RESULTS
        std::string file_name_y_predict_sd = results_folder + "/y_predict_sd_ntFit" + to_string(nt_fit) + "_ntPred" + to_string(nt_pred) + "_tInit" + to_string(nt_init_fit) + ".txt";
        write_vector(file_name_y_predict_sd, projMargSd, no_all);
#endif

#if 0
       MatrixXd Qinv_full(n,n);
       fun->compute_fullInverseQ(theta, Qinv_full);
       Vect temp = QinvSp.diagonal() - marg;
       Vect temp2 = Qinv_full.diagonal() - marg;
       //std::cout << "norm(diag(QinvBlks) - marg) = " << temp.norm() << std::endl;
       std::cout << "norm(marg) = " << marg.norm() << ", norm(diag(Qinv_full)) = " << Qinv_full.diagonal().norm() << std::endl;
       std::cout << "norm(diag(Qinv_full) - marg) = " << temp2.norm() << std::endl;
       std::cout << "norm(diag(Qinv_full) - marg) = " << temp2.head(50).transpose() << std::endl;

       MatrixXd Id(n,n);
       Id.setIdentity();
       SpMat Q(n,n);
       fun->construct_Q(theta, Q);
       std::cout << "norm(Qinv_full*Q - Id) = " << (Qinv_full*Q - Id).norm() << std::endl;

       //std::cout << "Qinv_full - QinvSp: " << Qinv_full.block(0,0,20,20) - QinvSp.block(0,0,20,20) << std::endl;

       MatrixXd QinvProjFULL = Ax_all*Qinv_full*Ax_allT;
       Vect projMargVarFULL = QinvProjFULL.diagonal();
       std::cout << "est. std dev 1st 10 loc FULL       : " << projMargVarFULL.head(20).cwiseSqrt().transpose() << std::endl;
#endif


#endif // end predict       


     } // end (MPI_rank == 0) for marginal variances

    // =================================== print times =================================== //
#if 1 

    int total_fn_call = fun->get_fct_count();

    t_total +=omp_get_wtime();
    if(MPI_rank == 0){
        // total number of post_theta_eval() calls 
        std::cout << "\ntotal number fn calls        : " << total_fn_call << std::endl;
        std::cout << "\ntime BFGS solver             : " << time_bfgs << " sec" << std::endl;
        std::cout << "time get covariance          : " << t_get_covariance << " sec" << std::endl;
        std::cout << "time get marginals FE        : " << t_get_marginals << " sec" << std::endl;
        std::cout << "total time                   : " << t_total << " sec" << std::endl;
    }

#endif

    delete fun;

#endif

    MPI_Finalize();
    return 0;
    
}
