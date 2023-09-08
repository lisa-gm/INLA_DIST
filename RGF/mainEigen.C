#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <unsupported/Eigen/SparseExtra>   // includes saveMarket

#include <armadillo>
#include "generate_testMat_selInv.cpp"
#include "../read_write_functions.cpp"
#include "helper_functions.h"

#include "RGF.H"

using Eigen::VectorXd;
using Eigen::MatrixXd;

typedef Eigen::VectorXd Vect;

//#define PRINT_MSG

#if 0
typedef CPX T;
#define assign_T(val) CPX(val, 0.0)
#else
typedef double T;
#define assign_T(val) val
#endif

void construct_Q_spatial(SpMat& Qs, Vect& theta, SpMat& c0, SpMat& g1, SpMat& g2){

	// Qs <- g[1]^2*Qgk.fun(sfem, g[2], order)
	// return(g^4 * fem$c0 + 2 * g^2 * fem$g1 + fem$g2)
	double exp_theta1 = exp(theta[1]);
	double exp_theta2 = exp(theta[2]);
	//double exp_theta1 = -3;
	//double exp_theta2 = 1.5;

	Qs = pow(exp_theta1,2)*(pow(exp_theta2, 4) * c0 + 2*pow(exp_theta2,2) * g1 + g2);

	#ifdef PRINT_MSG
		/*std::cout << "pow(exp_theta1,2) : \n" << pow(exp_theta1,2) << std::endl;
		std::cout << "pow(exp_theta2,2) : \n" << pow(exp_theta2,2) << std::endl;
		std::cout << "c0 : \n" << c0.block(0,0,10,10) << std::endl;
        std::cout << "g1 : \n" << g1.block(0,0,10,10) << std::endl;
        std::cout << "g2 : \n" << g2.block(0,0,10,10) << std::endl;*/
    #endif

	// extract triplet indices and insert into Qx
} 


void construct_Q_spat_temp(SpMat& Qst, Vect& theta, SpMat& c0, SpMat& g1, SpMat& g2, SpMat& g3,\
									  SpMat& M0, SpMat& M1, SpMat& M2){

	//std::cout << "theta : " << theta.transpose() << std::endl;

	double exp_theta1 = exp(theta[1]);
	double exp_theta2 = exp(theta[2]);
	double exp_theta3 = exp(theta[3]);

	/*double exp_theta1 = exp(-5.594859);
	double exp_theta2 = exp(1.039721);
	double exp_theta3 = exp(3.688879);*/

	//std::cout << "exp(theta) : " << exp(theta[0]) << " " << exp_theta1 << " " << exp_theta2 << " " << exp_theta3 << " " << std::endl;	

	// g^2 * fem$c0 + fem$g1
	SpMat q1s = pow(exp_theta2, 2) * c0 + g1;

	 // g^4 * fem$c0 + 2 * g^2 * fem$g1 + fem$g2
    SpMat q2s = pow(exp_theta2, 4) * c0 + 2 * pow(exp_theta2,2) * g1 + g2;

    // g^6 * fem$c0 + 3 * g^4 * fem$g1 + 3 * g^2 * fem$g2 + fem$g3
    SpMat q3s = pow(exp_theta2, 6) * c0 + 3 * pow(exp_theta2,4) * g1 + 3 * pow(exp_theta2,2) * g2 + g3;

    #ifdef PRINT_MSG
        /*std::cout << "theta u : " << exp_theta1 << " " << exp_theta2 << " " << exp_theta3 << std::endl;
    std::cout << "pow(exp_theta1,2) : \n" << pow(exp_theta1,2) << std::endl;
    std::cout << "pow(exp_theta2,2) : \n" << pow(exp_theta2,2) << std::endl;
    std::cout << "q1s : \n" << q1s.block(0,0,10,10) << std::endl;
    std::cout << "q2s : \n" << q2s.block(0,0,10,10) << std::endl;
    std::cout << "q3s : \n" << q3s.block(0,0,10,10) << std::endl;*/
    #endif

    // assemble overall precision matrix Q.st
    Qst = pow(exp_theta1,2)*(KroneckerProductSparse<SpMat, SpMat>(M0, q3s) + exp_theta3 *KroneckerProductSparse<SpMat, SpMat>(M1, q2s) + pow(exp_theta3, 2)* KroneckerProductSparse<SpMat, SpMat>(M2, q1s));

    //std::cout << "Qst : \n" << Qst.block(0,0,10,10) << std::endl;
}

#if 1
void construct_Q(SpMat& Q, int ns, int nt, int nb, Vect& theta, SpMat& c0, SpMat& g1, SpMat& g2, SpMat& g3,\
									  SpMat& M0, SpMat& M1, SpMat& M2, SpMat& Ax){

	double exp_theta0 = exp(theta[0]);
	int nu = ns*nt;

	SpMat Q_b = 1e-5*Eigen::MatrixXd::Identity(nb, nb).sparseView(); 
	/*std::cout << "Q_b " << std::endl;
	std::cout << Eigen::MatrixXd(Q_b) << std::endl;*/

	if(ns > 0){
		SpMat Qu(nu, nu);
		// TODO: find good way to assemble Qx

		if(nt > 1){
			construct_Q_spat_temp(Qu, theta, c0, g1, g2, g3, M0, M1, M2);
		} else {	
			construct_Q_spatial(Qu, theta, c0, g1, g2);
		}	

		//Qub0 <- sparseMatrix(i=NULL,j=NULL,dims=c(nb, ns))
		// construct Qx from Qs values, extend by zeros 
		size_t n = ns*nt + nb;
		SpMat Qx(n,n);         // default is column major			

		int nnz = Qu.nonZeros();
		Qx.reserve(nnz);

		for (int k=0; k<Qu.outerSize(); ++k)
		  for (SparseMatrix<double>::InnerIterator it(Qu,k); it; ++it)
		  {
		    Qx.insert(it.row(),it.col()) = it.value();                 
		  }

		//Qs.makeCompressed();
		//SpMat Qx = Map<SparseMatrix<double> >(ns+nb,ns+nb,nnz,Qs.outerIndexPtr(), // read-write
        //                   Qs.innerIndexPtr(),Qs.valuePtr());

		for(int i=nu; i<(n); i++){
			Qx.coeffRef(i,i) = 1e-5;
		}

		Qx.makeCompressed();

		#ifdef PRINT_MSG
			//std::cout << "Qx : \n" << Qx.block(0,0,10,10) << std::endl;
			//std::cout << "Ax : \n" << Ax.block(0,0,10,10) << std::endl;
		#endif

		Q =  Qx + exp_theta0 * Ax.transpose() * Ax;

		#ifdef PRINT_MSG
			std::cout << "exp(theta0) : " << exp_theta0 << std::endl;
			std::cout << "Qx dim : " << Qx.rows() << " " << Qx.cols() << std::endl;

			std::cout << "Q  dim : " << Q.rows() << " "  << Q.cols() << std::endl;
			//std::cout << "Q : \n" << Q.block(0,0,10,10) << std::endl;
			std::cout << "theta : \n" << theta.transpose() << std::endl;

		#endif
	}

	/*std::cout << "Q -  exp(theta)*B'*B " << std::endl;
	std::cout << Eigen::MatrixXd(*Q) - exp_theta*B.transpose()*B << std::endl;*/

}
#endif

void construct_Q(SpMat& Q, int ns, int nt, int nss, int nb, Vect& theta, SpMat& c0, SpMat& g1, SpMat& g2, SpMat& g3,\
									  SpMat& M0, SpMat& M1, SpMat& M2, SpMat& Ax){

    int n = ns*nt + nss + nb;
    SpMat Qx(n,n);

    if(nss == 0){
        SpMat Qst(ns*nt, ns*nt);
        construct_Q_spat_temp(Qst, theta, c0, g1, g2, g3, M0, M1, M2);
		
		int nnz = Qst.nonZeros();
		Qx.reserve(nnz);

		for (int k=0; k<Qst.outerSize(); ++k){
		  for (SparseMatrix<double>::InnerIterator it(Qst,k); it; ++it)
		  {
		    Qx.insert(it.row(),it.col()) = it.value();                 
		  }
        }

        printf("here.\n");
    } else {
        SpMat Qst(ns*nt, ns*nt);
        std::cout << "theta:           " << theta.transpose() << std::endl;
        std::cout << "theta(seq(0,3)): " << theta(seq(0,3)).transpose() << std::endl;
        Vect theta_spat_temp = theta(seq(0,3));
        construct_Q_spat_temp(Qst, theta_spat_temp, c0, g1, g2, g3, M0, M1, M2);

        size_t nnz_Qst = Qst.nonZeros();
        Qx.reserve(nnz_Qst);

        for (int k=0; k<Qst.outerSize(); ++k){
            for (SparseMatrix<double>::InnerIterator it(Qst,k); it; ++it)
            {
                Qx.insert(it.row(),it.col()) = it.value();                 
            }
        }

        SpMat Qs(nss, nss);
        // need to be careful about what theta values are accessed!! now dimension larger
        Vect theta_spat = theta(seq(3,5));
        construct_Q_spatial(Qs, theta_spat, c0, g1, g2);

        // insert entries of Qs
        for (int k=0; k<Qs.outerSize(); ++k){
            for (SparseMatrix<double>::InnerIterator it(Qs,k); it; ++it)
            {
            Qx.insert(it.row()+ns*nt,it.col()+ns*nt) = it.value();                 
            }
        }

    }

    std::cout << "dim(Ax) = " << Ax.rows() << " " << Ax.cols() << ", dim(Qx) = " << Qx.rows() << " " << Qx.cols() << std::endl;

    printf("here now. nb = %d, n = %d\n", nb, n);

    for(int i=ns*nt+nss; i < n; i++){
        printf("i = %d\n", i);
		// CAREFUL 1e-3 is arbitrary choice!!
		Qx.insert(i,i) = 1e-3;
	}	

    Qx.makeCompressed();

    double exp_theta0 = exp(theta[0]);
    Q = Qx + exp_theta0 * Ax.transpose()*Ax;

    printf("after Q\n");

}


// construct sparse Matrix from invBlks 
// invBlks have particular order ... non-contiguous ... 
// but we want to fill sparse matrix by column 
void construct_spInvQBlks(size_t ns, size_t nt, size_t nb, size_t nnz_invBlks, T* invBlks, SpMat& QinvBlks){

    //std::cout << "nnz_invBlks = " << nnz_invBlks << ", ns = " << ns << ", nt = " << nt << ", nb = " << nb << std::endl;
    size_t n = ns*nt + nb;
    SpMat QinvBlks_lower(n,n);
    QinvBlks_lower.reserve(nnz_invBlks);

    // deal with final block later
    for(int col=0; col<ns*nt; col++){
        int ts = col / ns; // determine which time step we are in
        int col_ts = col % ns; // col relative to the current timestep
        //printf("ts = %d, col ts = %d\n", ts, col_ts);
        // row 
        for(int i=0; i<ns; i++){
            int row = ns * ts + i;
            // get correct entry from invBlks array
            // invBlks sorted as D1, F1, D2, F2, ... Fn, Dn+1n+1, column-major
            int ind_invBlks = ts * (ns + nb) * ns + col_ts * ns + i;
            QinvBlks_lower.insert(row, col) = invBlks[ind_invBlks];

        }

        for(int i=0; i<nb; i++){   
            // always last rows
            int row = ns*nt+i;
            int ind_invBlks = ts * (ns + nb) * ns + ns * ns + col_ts * nb + i;
            QinvBlks_lower.insert(row, col) = invBlks[ind_invBlks];

        }
    }

    // ... and finally for last nb columns
    int ind_offset = (ns + nb) * ns * nt;
    for(int col=ns*nt; col<ns*nt+nb; col++){
        int col_ts = col % ns;
        //printf("col ts = %d\n", col_ts);
        for(int i=0; i<nb; i++){
            int row = ns*nt+i;
            int ind_invBlks = ind_offset + col_ts*nb + i;
            QinvBlks_lower.insert(row, col) = invBlks[ind_invBlks];
        }
    }

    QinvBlks = QinvBlks_lower.selfadjointView<Lower>();

}


// construct sparse Matrix from invBlks 
// invBlks have particular order ... non-contiguous ... 
// but we want to fill sparse matrix by column 
void construct_lower_CSC_invBlks(size_t ns, size_t nt, size_t nb, size_t nnz_lower_invBlks, T* invBlks, SpMat& QinvBlks){

    //std::cout << "nnz_invBlks lower = " << nnz_lower_invBlks << ", ns = " << ns << ", nt = " << nt << ", nb = " << nb << std::endl;
    int n = ns*nt + nb;
    SpMat QinvBlks_lower(n,n);
    QinvBlks_lower.reserve(nnz_lower_invBlks);

    int* row_ind_a; // row index of each nnz value
    int* col_ptr_a; // list of val indices where each column starts
    T* a;

    row_ind_a = new int [nnz_lower_invBlks];
    col_ptr_a = new int [n+1];
    a         = new T[nnz_lower_invBlks];

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
void construct_full_CSC_invBlks(size_t ns, size_t nt, size_t nb, size_t nnz_invBlks, T* invBlks, SpMat& QinvBlks){

    //std::cout << "nnz_invBlks lower = " << nnz_lower_invBlks << ", ns = " << ns << ", nt = " << nt << ", nb = " << nb << std::endl;
    int n = ns*nt + nb;
    //SpMat QinvBlks_lower(n,n);
    QinvBlks.reserve(nnz_invBlks);

    int* row_ind_a; // row index of each nnz value
    int* col_ptr_a; // list of val indices where each column starts
    T* a;

    row_ind_a = new int [nnz_invBlks];
    col_ptr_a = new int [n+1];
    a         = new T[nnz_invBlks];

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

#if 0
    printf("row_ind_a: ");
    for(int i=0; i<nnz_invBlks; i++){
        printf("%d ", row_ind_a[i]);
    }
    printf("\n");


    printf("col_ptr_a: ");
    for(int i=0; i<n+1; i++){
        printf("%d ", col_ptr_a[i]);
    }
    printf("\n");

    printf("a: ");
    for(int i=0; i<nnz_invBlks; i++){
        printf("%f ", a[i]);
    }
    printf("\n");
#endif

    t_loop = - omp_get_wtime();
    QinvBlks =  Eigen::Map<Eigen::SparseMatrix<double> >(n,n,nnz_invBlks,col_ptr_a, row_ind_a,a);
    t_loop += omp_get_wtime();
    printf("fullCSC: time Eigen Map : %f\n", t_loop);

    //std::cout << "QinvBlks full: \n" << MatrixXd(QinvBlks) << std::endl;

}


// compute variances for timestep ts
// Ax column-major
void compute_marginals_ts(size_t ns, size_t nt, size_t nb, size_t ts, T* invBlks, SpMat& Ax, Vect& variance_vec){
    
    //int nnz_subQ = ns*ns + 
    //SpMat Q()
}

/* ===================================================================== */

int main(int argc, char* argv[])
{

size_t i; // iteration variable

#if 0

    /*
    int ns=1;
    int nt=6;
    int nb=1;
    int no=0;

    int n = ns*nt+nb;

    //SpMat Q = gen_test_mat_base1();
    SpMat Q = gen_test_mat_base2();
    std::cout << "Q: \n" << Q << std::endl;
    */

    int ns=3;
    int nss=0;
    int nt=5;
    int nb=3;
    int n = ns*nt + nb;

    SpMat Q = gen_test_mat_base3(ns, nt, nb);


    Vect rhs(n);
    rhs.setOnes(n);



#else

    if(argc != 1 + 7){
        std::cout << "wrong number of input parameters. " << std::endl;

        std::cerr << "INLA Call : ns nt nss nb no path/to/files solver_type" << std::endl;

        std::cerr << "[integer:ns]                number of spatial grid points " << std::endl;
        std::cerr << "[integer:nt]                number of temporal grid points " << std::endl;
        std::cerr << "[integer:nss]               number of spatial grid points of ADD. SPATIAL FIELD " << std::endl;
        std::cerr << "[integer:nb]                number of fixed effects" << std::endl;
        std::cerr << "[integer:no]                number of data samples" << std::endl;

        std::cerr << "[string:base_path]          path to folder containing matrix files " << std::endl;

        std::cerr << "[string:solver_type]        BTA or PARDISO" << std::endl;
    

        exit(1);
    }

    std::cout << "reading in example. " << std::endl;

    size_t ns = atoi(argv[1]);
    size_t nt = atoi(argv[2]);
    size_t nss = atoi(argv[3]);
    size_t nb = atoi(argv[4]);
    std::cout << "ns = " << ns << ", nt = " << nt << ", nss = " << nss << ", nb = " << nb << std::endl;
    //size_t no = atoi(argv[4]);
    std::string no_s = argv[5];
    // to be filled later
    size_t no;

    // set nt = 1 if ns > 0 & nt = 0
    if(ns > 0 && nt == 0){
        nt = 1;
    } 

    // also save as string
    std::string ns_s = std::to_string(ns);
    std::string nt_s = std::to_string(nt);
    std::string nss_s = std::to_string(nss);
    std::string nb_s = std::to_string(nb);
    //std::string no_s = std::to_string(no); 
    std::string n_s  = std::to_string(ns*nt + nss + nb);

    std::string base_path = argv[6];    

    std::string solver_type = argv[7];
    // check if solver type is neither PARDISO nor BTA :
    if(solver_type.compare("PARDISO") != 0 && solver_type.compare("BTA") != 0){
        std::cout << "Unknown solver type. Available options are :\nPARDISO\nBTA" << std::endl;
        exit(1);
    }

    /* ---------------- read in matrices ---------------- */

    // dimension hyperparamter vector
    int dim_th;

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
    SpMat Ax; 
    Vect y;

    if(ns == 0 && nt == 0){

        dim_th = 1;

        // read in design matrix 
        // files containing B
        std::string B_file        =  base_path + "/B_" + no_s + "_" + nb_s + ".dat";
        file_exists(B_file); 

        // casting no_s as integer
        no = std::stoi(no_s);
        std::cout << "total number of observations : " << no << std::endl;
      
        B = read_matrix(B_file, no, nb);

        // std::cout << "y : \n"  << y << std::endl;    
        // std::cout << "B : \n" << B << std::endl;

    } else if(ns > 0 && nt == 1 && nss == 0){

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
        std::string Ax_file     =  base_path + "/Ax_" + no_s + "_" + n_s + ".dat";
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
        std::cout << "total number of observations : " << no << std::endl;


        /*std::cout << "g1 : \n" << g1.block(0,0,10,10) << std::endl;
        std::cout << "g2 : \n" << g2.block(0,0,10,10) << std::endl;
        std::cout << "Ax : \n" << Ax.block(0,0,10,10) << std::endl;*/

    } else if(ns > 0 && nt > 1){

        if(nss == 0){
            dim_th = 4;
        } else {
            dim_th = 6;
        }

        printf("spatial-temporal model");
        if(nss > 0){
            printf(" with add. spatial field");
        }
        printf(".\n");
    

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

        // check projection matrix for A.st
        std::string Ax_file     =  base_path + "/Ax_" + no_s + "_" + n_s + ".dat";
        file_exists(Ax_file);

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

        Ax = readCSC(Ax_file);
        // get rows from the matrix directly
        // doesnt work for B
        no = Ax.rows();

        //std::cout << "total number of observations : " << no << std::endl;
        std::cout << "read in all matrices." << std::endl;

    } else {
        std::cout << "invalid parameters : ns nt nss !!" << std::endl;
        exit(1);
    }

    // data y
    std::string y_file        =  base_path + "/y_" + no_s + "_1" + ".dat";
    file_exists(y_file);
    // at this point no is set ... 
    // not a pretty solution. 
    y = read_matrix(y_file, no, 1);

    /* ----------------------- initialise random theta -------------------------------- */

    Vect theta(dim_th);
    Vect theta_prior(dim_th);

	if(nt == 1){
	    theta << -1.5,-5,-2;
	    //theta.print();
    } else if(nt > 1 && nss == 0){
        //theta << 5, -10, 2.5, 1;
        //theta << 4.000000, -3.344954,  1.039721,  1.386294; // equals 4,0,0,0 in param scale     
        theta << -1.998039, -9.828957,  1.981187,  8.288427;   
        std::cout << "theta : " << theta.transpose() << std::endl;
	    //theta = {3, -5, 1, 2};
	    //theta.print();
  	} else {
        theta << 1.386796, -4.434666, 0.6711493, 1.632289, -5.058083, 2.664039;
  	}


printf("# threads: %d\n", omp_get_max_threads());

//#pragma omp parallel
//{

//if(omp_get_thread_num() == 0) // instead of #pragma omp task -> want always the same thread to do same task
//{

#if 0

    int nx = ns*nt + nss;

    SpMat Qx(nx, nx);

    double t_Qx_factorise;
    RGF<T> *solver_Qx;
    solver_Qx = new RGF<T>(ns, nt, nss);

    double log_det_Qx;

    for(int c=0; c<1; c++){
        //theta = theta + Vect::Random(theta.size());
        std::cout << "\niter = " << c << ". Constructing precision matrix Qx. theta : " << theta.transpose() << std::endl;

        if(nss == 0){
            construct_Q_spat_temp(Qx, theta, c0, g1, g2, g3, M0, M1, M2);
        } else {
            SpMat Qst(ns*nt, ns*nt);
            std::cout << "theta:           " << theta.transpose() << std::endl;
            std::cout << "theta(seq(0,3)): " << theta(seq(0,3)).transpose() << std::endl;
            Vect theta_spat_temp = theta(seq(0,3));
            construct_Q_spat_temp(Qst, theta_spat_temp, c0, g1, g2, g3, M0, M1, M2);

            size_t nnz_Qst = Qst.nonZeros();
            Qx.reserve(nnz_Qst);

            for (int k=0; k<Qst.outerSize(); ++k){
                for (SparseMatrix<double>::InnerIterator it(Qst,k); it; ++it)
                {
                    Qx.insert(it.row(),it.col()) = it.value();                 
                }
            }

            SpMat Qs(nss, nss);
            // need to be careful about what theta values are accessed!! now dimension larger
            Vect theta_spat = theta(seq(3,5));
            construct_Q_spatial(Qs, theta_spat, c0, g1, g2);

            // insert entries of Qs
            for (int k=0; k<Qs.outerSize(); ++k){
                for (SparseMatrix<double>::InnerIterator it(Qs,k); it; ++it)
                {
                Qx.insert(it.row()+ns*nt,it.col()+ns*nt) = it.value();                 
                }
            }

        }

        //SpMat epsId(nx,nx);
        //epsId.setIdentity();
        //epsId = 1e-4*epsId;
        //Qx = Qx + epsId;

        // only take lower triangular part of A
        SpMat Qx_lower = Qx.triangularView<Lower>(); 
        size_t nnz_Qx  = Qx_lower.nonZeros();

        /*std::string Qx_fileName = "Qx_" + to_string(n) + ".txt";
        write_sym_CSC_matrix(Qx_fileName, Qx_lower);
        exit(1);*/

#if 0
        //std::string Qx_lower_file = "Qst_lower_ns" + ns_s + "_nt" + nt_s + "_nb0_" + to_string(nx) + "_" + to_string(nx) + ".mtx";
        //std::string Qx_firstBlock_file = "Qst_firstBlock_" + to_string(ns) + "_" + to_string(ns) + ".txt";
        //MatrixXd Qx_first = MatrixXd(Qx.block(0,0,ns,ns));
        //write_matrix(Qx_firstBlock_file, Qx_first);
        SpMat Qx_first = Qx.block(0,0,ns,ns);
        std::cout << "nnz(Q) = " << Qx_first.nonZeros() << std::endl;
        //write_sym_CSC_matrix(Qx_firstBlock_file, Qx_first);
        std::string Qx_firstBlock_file = "Qst_firstBlock_" + to_string(ns) + "_" + to_string(ns) + ".mtx";
        Eigen::saveMarket(Qx_first, Qx_firstBlock_file);
        exit(1);
#endif

        Qx_lower.makeCompressed();

        size_t* ia_Qx; 
        size_t* ja_Qx;
        T* a_Qx; 

        // allocate memory
        ia_Qx = new long unsigned int [nx+1];
        ja_Qx = new long unsigned int [nnz_Qx];
        a_Qx  = new double [nnz_Qx];

        for (i = 0; i < nx+1; ++i){
            ia_Qx[i] = Qx_lower.outerIndexPtr()[i]; 
        }  

        for (i = 0; i < nnz_Qx; ++i){
            ja_Qx[i] = Qx_lower.innerIndexPtr()[i];
        }  

        for (i = 0; i < nnz_Qx; ++i){
            a_Qx[i] = Qx_lower.valuePtr()[i];
        }

        t_Qx_factorise = get_time(0.0);
        //solver->solve_equation(GR);
        //double flops_Qx_factorize = solver_Qx->factorize_noCopyHost(ia_Qx, ja_Qx, a_Qx, log_det_Qx);
        //printf("after factorize no copy to host.\n");
        
        double flops_Qx_factorize = solver_Qx->factorize(ia_Qx, ja_Qx, a_Qx);
        log_det_Qx = solver_Qx->logDet(ia_Qx, ja_Qx, a_Qx);

        t_Qx_factorise = get_time(t_Qx_factorise);

        printf("logdet       : %f\n", log_det_Qx);
        printf("time chol(Qx): %lg\n",t_Qx_factorise);

        T *invDiag_Qx = new T[nx];
        solver_Qx->RGFdiag(ia_Qx, ja_Qx, a_Qx, invDiag_Qx);

        Vect invDiag_Qx_vec(nx);
        for(int i=0; i<nx; i++){
            invDiag_Qx_vec[i] = invDiag_Qx[i];
        }

        printf("norm(invDiag_Qx)      : %f\n", invDiag_Qx_vec.norm());
        
        T *invQa = new T[Qx_lower.nonZeros()];
        solver_Qx->RGFselInv(ia_Qx, ja_Qx, a_Qx, invQa);

        SpMat invQx_lower = Eigen::Map<Eigen::SparseMatrix<double> >(nx,nx,Qx_lower.nonZeros(),Qx_lower.outerIndexPtr(), // read-write
                               Qx_lower.innerIndexPtr(),invQa);

        // TODO: more efficient way to do this?
        //SpMat invQx_new = invQ_new_lower.selfadjointView<Lower>();

        printf("norm(invDiag_full_Qx) : %f\n", invQx_lower.diagonal().norm());


        std::cout << "norm(diag(invQ_new) - diag(invDiag)) = " << (invQx_lower.diagonal() - invDiag_Qx_vec).norm() << std::endl;


        //Vect invDiag_Qx(nx);
        //solver_Qx->RGFdiag(ia_Qx, ja_Qx, a_Qx, invDiag_Qx);



        delete[] ia_Qx;
        delete[] ja_Qx;
        delete[] a_Qx;

        delete[] invDiag_Qx;
        delete[] invQa;

    }

    delete solver_Qx;

#endif // end construct Qx


#endif // end dummy example or reading in matrices

//} // end omp if(thread 0)


//if(omp_get_thread_num() == 1 || omp_get_num_threads() < 2)
//{

#if 1


    int n = ns*nt + nss + nb;
    SpMat Q(n,n);
    Vect rhs(n);
    double exp_theta = exp(theta[0]);
    rhs = exp_theta*Ax.transpose()*y;
    std::cout << "\nConstructing precision matrix Qxy. " << std::endl;
    //std::cout << "Setting Ax to zero." << std::endl;
    //Ax.makeCompressed();
    //Ax.setZero();
    //for(int c=0; c<1; c++){
        //theta = theta + Vect::Random(theta.size());
        //std::cout << "\niter = " << c << ". Constructing precision matrix Qxy. theta : " << theta.transpose() << std::endl;   
        
        construct_Q(Q, ns, nt, nss, nb, theta, c0, g1, g2, g3, M0, M1, M2, Ax); 
        std::cout << "Q : \n" << Q.block(0,0,8,8) << std::endl;

        //SpMat epsId(n,n);
        //epsId.setIdentity();
        //epsId = 1e-4*epsId;

        //Q = Q + epsId;
    
    	// =========================================================================== //
    	std::cout << "Converting Eigen Matrices to CSR format. " << std::endl;

    	// only take lower triangular part of A
        SpMat Q_lower = Q.triangularView<Lower>(); 
        size_t nnz    = Q_lower.nonZeros();

        printf("nnz(Q_lower) = %ld\n", nnz);

        /*
        //SpMat Q_lower_fstB = Q_lower.block(0,0,ns,ns);
        //std::string filename =  "Qst_firstBlock_lower_" + to_string(ns) + "_" + to_string(ns) + ".mtx";
        std::string filename =  "Q_lower_n" + to_string(n) + "_ns" + to_string(ns) + "_nt" + to_string(nt) + "_nb" + to_string(nb) + ".mtx";
        Eigen::saveMarket(Q_lower, filename);
        exit(1);
        */

        //std::string Q_fileName = "Q_" + to_string(n) + ".txt";
        //write_sym_CSC_matrix(Q_fileName, Q_lower);
        
        size_t* ia; 
        size_t* ja;
        T* a; 
        T *b;
      	T *x;

      	b        = new T[n];
      	x        = new T[n];

        // allocate memory
        ia = new long unsigned int [n+1];
        ja = new long unsigned int [nnz];
        a  = new double [nnz];

        Q_lower.makeCompressed();

        for (i = 0; i < n+1; ++i){
            ia[i] = Q_lower.outerIndexPtr()[i]; 
        }  

        for (i = 0; i < nnz; ++i){
            ja[i] = Q_lower.innerIndexPtr()[i];
        }  

        for (i = 0; i < nnz; ++i){
            a[i] = Q_lower.valuePtr()[i];
        }

        double t_factorise;
    	double t_solve;

        // *** pin GPU & combine with appropriate cores *** //
        int GPU_rank = 0;
        cudaSetDevice(GPU_rank);
        int numa_node = topo_get_numNode(GPU_rank);

        int* hwt = NULL;
        int hwt_count = read_numa_threads(numa_node, &hwt);
        pin_hwthreads(1, &hwt[omp_get_thread_num()]);
        std::cout<<"Pinning GPU & hw threads. GPU rank : "<<GPU_rank <<", tid: "<<omp_get_thread_num()<<", NUMA domain ID: "<<numa_node;
        std::cout<<", hwthreads: " << hwt[omp_get_thread_num()] << std::endl;
        // *********************************************** //

        RGF<T> *solver;
        solver = new RGF<T>(ns, nt, nss+nb, GPU_rank);

        int m = 1;
        Vect t_factorize_vec(m-1);
        double log_det;

        for(int i=0; i<m; i++){

            printf("i = %d\n", i);
            t_factorise = get_time(0.0);
            //solver->solve_equation(GR);
            double flops_factorize = solver->factorize(ia, ja, a);
            log_det = solver->logDet(ia, ja, a);
            printf("logdet: %f\n", log_det);
            t_factorise = get_time(t_factorise);
            printf("time factorize:             %f\n", t_factorise);

            if(i>0){
                t_factorize_vec(i-1) = t_factorise;
            }

            t_solve = get_time(0.0); 
            double flops_solve = solver->solve(ia, ja, a, x, b, 1);
            t_solve = get_time(t_solve);
            //printf("flops solve:     %f\n", flops_solve);

            //printf("time chol(Q): %lg\n",t_factorise);
            printf("time solve:                %f\n",t_solve);

            t_factorise = get_time(0.0);
            flops_factorize = solver->factorize_noCopyHost(ia, ja, a, log_det);
            t_factorise = get_time(t_factorise);
            printf("time factorize noCopyHost: %f\n", t_factorise);

            printf("logdet: %f\n", log_det);

            // assign b to correct format
            /*for (int i = 0; i < n; i++){
                b[i] = rhs[i];
                //printf("%f\n", b[i]);
            }*/

            printf("Residual norm.:           %e\n", solver->residualNorm(x, b));
            printf("Residual norm normalized: %e\n", solver->residualNormNormalized(x, b));

        }

        //std::cout << "factorize times: " << t_factorize_vec.transpose() << std::endl;


    //}

  	// create file with solution vector
  	/*
    std::string sol_x_file_name = "x_sol_RGF_ns" + ns_s + "_nt" + nt_s + "_nb" + nb_s + "_no" + no_s +".dat";
  	std::ofstream sol_x_file(sol_x_file_name,    std::ios::out | std::ios::trunc);

	for (i = 0; i < n; i++) {
		sol_x_file << x[i] << std::endl;
		// sol_x_file << x[i] << std::endl; 
	}

  sol_x_file.close();
  */

#if 0
    // true inv diag from Eigen
    //SimplicialLLT<SpMat, Eigen::Lower, Eigen::NaturalOrdering<int>> solverQ;
    SimplicialLLT<SpMat> solverQ;
    solverQ.compute(Q);

   if(solverQ.info()!=Success) {
     cout << "Oh: Very bad" << endl;
   }

   SpMat L = solverQ.matrixL();
   /*if(n < 20){
        std:cout << "L: \n" << MatrixXd(L) << std::endl;
    }*/

   SpMat eye(n,n);
   eye.setIdentity();

   // compute log sum by hand
   double logDetEigen = 0.0;
   for(int i = 0; i<n; i++){
        logDetEigen += log(L.coeff(i,i));
   }
   logDetEigen *=2.0;

   //std::cout << "diag(L Eigen) : " << L.diagonal().transpose() << std::endl;
   std::cout << "log Det Eigen : " << logDetEigen << std::endl;
   std::cout << "diff Log Dets : " << logDetEigen - log_det << std::endl;

   SpMat inv_Q = solverQ.solve(eye);
   if(n < 25){
    MatrixXd inv_Q_dense = MatrixXd(inv_Q.triangularView<Lower>());
    std::cout << "inv(Q)\n" << inv_Q_dense << std::endl;
   }

#endif


#if 1

    T *invDiag;
    invDiag  = new T[n];

    double t_invDiag;
    t_invDiag = get_time(0.0);
    double flops_invDiag = solver->RGFdiag(ia, ja, a, invDiag);
    t_invDiag = get_time(t_invDiag);
    double log_detRGFdiag = solver->logDet(ia, ja, a);

    if(n < 25){
        printf("\nRGFinvDiag: ");
        for(i=0; i<n; i++){
            printf(" %f", invDiag[i]);
        }
        printf("\n");
    }

    //printf("computed RGFdiag\n");

    //printf("flops inv:      %f\n", flops_invDiag);

    //solver->init_supernode()
    T* invQa;
    invQa = new T[nnz];
    //printf("before rgfselinv\n");
    double flops_invQa = solver->RGFselInv(ia, ja, a, invQa);
    //printf("before logDetselInv\n");
    double log_detRGFselInv = solver->logDet(ia, ja, a);

    if(n < 25){
        printf("invQa : ");
        for(int i=0; i<nnz; i++){
            printf(" %f", invQa[i]);
        }
        printf("\n");
    }

    //printf("computed RGFselInv\n");

    SpMat invQ_new_lower = Eigen::Map<Eigen::SparseMatrix<double> >(n,n,nnz,Q_lower.outerIndexPtr(), // read-write
                               Q_lower.innerIndexPtr(),invQa);


    if(n < 25){
        std::cout << "invQ_new:\n" << MatrixXd(invQ_new_lower) << std::endl;
    }

  // TODO: more efficient way to do this?
    SpMat invQ_new = invQ_new_lower.selfadjointView<Lower>();

    Vect invDiag_vec(n);
    for(int i=0; i<n; i++){
        invDiag_vec[i] = invDiag[i];
    }

    std::cout << "norm(diag(invQ_new)) = " << invQ_new.diagonal().norm() << std::endl;
    std::cout << "norm(invDiag)        = " << invDiag_vec.norm() << std::endl;   
    //std::cout << "norm(diag(invEigen)) = " << inv_Q.diagonal().norm() << std::endl;    
    std::cout << "norm(diag(invQ_new) - diag(invDiag)) = " << (invQ_new.diagonal() - invDiag_vec).norm() << std::endl;
    //std::cout << "norm(diag(invQ_new) - diag(invEigen)) = " << (invQ_new.diagonal() - inv_Q.diagonal()).norm() << std::endl;

    //std::string invQ_fileName = "invQ_del2CompStream_" + to_string(n) + ".txt";
    //write_sym_CSC_matrix(invQ_fileName, invQ_new_lower);

    /*
    std::string invQ_new_fileName = "invQ_new_diag_" + to_string(n) + ".txt";
    ofstream invQ_new_file(invQ_new_fileName,    ios::out | ::ios::trunc);

    std::string invQ_diag_fileName = "invQ_diag_" + to_string(n) + ".txt";
    ofstream invQ_diag_file(invQ_diag_fileName,    ios::out | ::ios::trunc);

    //std::string invQ_full_fileName = "invQ_full_diag_" + to_string(n) + ".txt";
    //ofstream invQ_full_file(invQ_full_fileName,    ios::out | ::ios::trunc);

    for(int i=0; i<n; i++){
        invQ_new_file << std::setprecision(7) << invQ_new.diagonal()[i] << endl;
        invQ_diag_file << std::setprecision(7) << invDiag_vec[i] << std::endl;
        //invQ_full_file << std::setprecision(7) << inv_Q.diagonal()[i] << endl;
    }

    invQ_new_file.close();
    invQ_diag_file.close();
    //invQ_full_file.close();

    */

    /*
    t_invBlks = get_time(0.0);
    double flops_invBlks = solver->RGFinvBlks(ia, ja, a, invBlks);
    t_invBlks= get_time(t_invBlks);
    std::cout << "diff Log Dets : " << log_detRGFdiag - log_detRGFselInv << std::endl;

    printf("RGF factorise time: %lg\n",t_factorise);
    printf("RGF solve     time: %lg\n",t_solve);
    printf("RGF inv Diag  time: %lg\n",t_invDiag);
    printf("RGF inv Blks  time: %lg\n",t_invBlks);
    */

    // now assemble invBlks to correct sparse matrix -> column major -> iterate through columns
    // careful with block structure, need to be alternating betwen diagonal & off diagonal dense blocks
#if 0
    SpMat QinvBlks_comp(n,n);
    QinvBlks_comp.reserve(ns*ns*nt+2*ns*nb*nt+nb*ns);

    double t_constrQinvBlks_comp = -omp_get_wtime();
    construct_spInvQBlks(ns, nt, nb, nnz_invBlks, invBlks, QinvBlks_comp);
    t_constrQinvBlks_comp += omp_get_wtime();
    printf("Assemble QinvBlks comp time: %lg\n",t_constrQinvBlks_comp);
#endif

    /*
    size_t nnz_lower_invBlks = nt*ns*(ns+1)/2 + ns*nb*nt + (nb+1)*nb/2;
    SpMat QinvBlks_comp(n,n);
    QinvBlks_comp.reserve(nnz_lower_invBlks);

    double t_constrQinvBlks = -omp_get_wtime();
    construct_lower_CSC_invBlks(ns, nt, nb, nnz_lower_invBlks, invBlks, QinvBlks_comp);
    t_constrQinvBlks += omp_get_wtime();
    printf("Assemble QinvBlks lower time: %lg\n \n",t_constrQinvBlks);
    
    SpMat QinvBlks(n,n);
    size_t nnz_full_invBlks = ns*ns*nt + 2*ns*nb*nt + nb*nb;
    double t_constrQinvBlks = -omp_get_wtime();
    construct_full_CSC_invBlks(ns, nt, nb, nnz_full_invBlks, invBlks, QinvBlks);
    t_constrQinvBlks += omp_get_wtime();
    printf("Assemble QinvBlks full time : %lg\n \n",t_constrQinvBlks);
    */

//#ifdef PRINT_MSG
    //std::cout << "diff(Q_inv_comp - QinvBlks) :\n" << MatrixXd(QinvBlks_comp - QinvBlks_full) << std::endl;

    //std::cout << "inv(Q):\n" << MatrixXd(inv_Q) << std::endl;
    //std::cout << "norm(QinvBlks_comp - QinvBlks) : " << (QinvBlks_comp - QinvBlks).norm() << std::endl;
//#endif

    // print/write diag 
    /*
    string sel_inv_file_name = "sel_inv_RGF_ns"+to_string(ns)+"_nt"+to_string(nt)+"_nb"+ to_string(nb) + "_no" + to_string(no) +".dat";
    cout << sel_inv_file_name << endl;
    ofstream sel_inv_file(sel_inv_file_name,    ios::out | ::ios::trunc);

    for (int i = 0; i < n; i++){
        sel_inv_file << invDiag[i] << endl;
    }

    sel_inv_file.close();
    cout << "after writing file " << endl;
    */

   /*
   Vect invDiag_vec(n);
    // assign b to correct format
    for (int i = 0; i < n; i++){
        invDiag_vec[i] = invDiag[i];
        //printf("%f\n", b[i]);
    }

    Vect invDiagfBlks(n);
    invDiagfBlks = QinvBlks.diagonal();

   //cout << "Q:\n" << Q << endl;

    //std::cout << "inv_Q - invQ_new :\n" << inv_Q - invQ_new << std::endl;

    printf("norm(invDiag - invDiagfBlks)   : %f\n", (invDiag_vec - invDiagfBlks).norm());
    cout << "norm(invQ_new - inv(Q))         : " << (invQ_new.diagonal() - inv_Q.diagonal()).norm() << std::endl;
    cout << "norm(diag(invQ))   : " << inv_Q.diagonal().norm() << std::endl;
    printf("norm(invDiagfBlks) : %f\n", invDiagfBlks.norm());
    printf("norm(invDiag)      : %f\n", invDiag_vec.norm());

    // to file
    std::string invQ_Eigen_fileName = "invQ_diag_Eigen.txt";
    ofstream invQ_Eigen_file(invQ_Eigen_fileName,    ios::out | ::ios::trunc);

    std::string invQ_selInv_fileName = "invQ_diag_selInv.txt";
     ofstream invQ_selInv_file(invQ_selInv_fileName,    ios::out | ::ios::trunc);

    for(int i=0; i<n; i++){
        invQ_Eigen_file << std::setprecision(15) << inv_Q.diagonal()[i] << endl;
        invQ_selInv_file << std::setprecision(15) << invQ_new.diagonal()[i] << std::endl;
    }

    invQ_Eigen_file.close();
    invQ_selInv_file.close();
    */

    /*
    MatrixXd Qinv_proj_fullInv = Ax * inv_Q * Ax.transpose();
    MatrixXd Qinv_proj         = Ax * invQ_new * Ax.transpose();

    std::cout << "diag(Qinv_proj_fullInv) : " << Qinv_proj_fullInv.diagonal().head(10).transpose() << std::endl;
    std::cout << "diag(Qinv)              : " << Qinv_proj.diagonal().head(10).transpose() << std::endl;
    MatrixXd temp = Qinv_proj_fullInv - Qinv_proj;
    std::cout << "norm(Qinv_proj - Qinv_proj_fullInv) : " << temp.diagonal().norm() << std::endl;
    */

    if(n < 20){
        //cout << "\ninvDiag BTA            : " << invDiag_vec.transpose() << std::endl;
        //cout << "invDiag from blks BTA  : "   << invDiagfBlks.transpose() << std::endl;
        //cout << "Eigen diag(inv_Q)      : "   << inv_Q.diagonal().transpose() << endl;
    } else {
#ifdef PRINT_MSG
        cout << "\ninvDiag BTA[1:20]       : " << invDiag_vec.head(20).transpose() << std::endl;
        //cout << "invDiag from blks BTA   : "   << invDiagBlks.transpose() << std::endl;
        //cout << "Eigen diag(inv_Q)[1:20] : " << inv_Q.diagonal().head(20).transpose() << endl;
        //cout << "norm(invDiag - inv(Q))  : " << (invDiag - inv_Q.diagonal()).norm() << std::endl;
#endif   
    }


  delete[] invDiag;

#endif

  
  // free memory
  delete solver;

  delete[] ia;
  delete[] ja;
  delete[] a;

  delete[] x;
  delete[] b;

  //} // end if

  #endif

  //} // end omp parallel


    
  return 0;


  }
