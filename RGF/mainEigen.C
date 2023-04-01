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
    int nt=50;
    int nb=2;
    int n = ns*nt + nb;

    SpMat Q = gen_test_mat_base3(ns, nt, nb);

    Vect rhs(n);
    rhs.setOnes(n);



#else

    if(argc != 1 + 6){
        std::cout << "wrong number of input parameters. " << std::endl;

        std::cerr << "INLA Call : ns nt nb no path/to/files solver_type" << std::endl;

        std::cerr << "[integer:ns]                number of spatial grid points " << std::endl;
        std::cerr << "[integer:nt]                number of temporal grid points " << std::endl;
        std::cerr << "[integer:nb]                number of fixed effects" << std::endl;
        std::cerr << "[integer:no]                number of data samples" << std::endl;

        std::cerr << "[string:base_path]          path to folder containing matrix files " << std::endl;

        std::cerr << "[string:solver_type]        RGF or PARDISO" << std::endl;
    

        exit(1);
    }

    std::cout << "reading in example. " << std::endl;

    size_t ns = atoi(argv[1]);
    size_t nt = atoi(argv[2]);
    size_t nb = atoi(argv[3]);
    std::cout << "ns = " << ns << ", nt = " << nt << ", nb = " << nb << std::endl;
    //size_t no = atoi(argv[4]);
    std::string no_s = argv[4];
    // to be filled later
    size_t no;

    // set nt = 1 if ns > 0 & nt = 0
    if(ns > 0 && nt == 0){
        nt = 1;
    } 

    // also save as string
    std::string ns_s = std::to_string(ns);
    std::string nt_s = std::to_string(nt);
    std::string nb_s = std::to_string(nb);
    //std::string no_s = std::to_string(no); 
    std::string n_s  = std::to_string(ns*nt + nb);

    std::string base_path = argv[5];    

    std::string solver_type = argv[6];
    // check if solver type is neither PARDISO nor RGF :
    if(solver_type.compare("PARDISO") != 0 && solver_type.compare("RGF") != 0){
        std::cout << "Unknown solver type. Available options are :\nPARDISO\nRGF" << std::endl;
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

    } else if(ns > 0 && nt == 1){

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

    } else if(ns > 0 && nt > 1) {

        std::cout << "spatial-temporal model. Reading in matrices." << std::endl;

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
        std::cout << "total number of observations : " << no << std::endl;

    } else {
        std::cout << "invalid parameters : ns nt !!" << std::endl;
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
  	} else {
	    //theta << 5, -10, 2.5, 1;
        theta << 4.000000, -3.344954,  1.039721,  1.386294; // equals 4,0,0,0 in param scale        
        std::cout << "theta : " << theta.transpose() << std::endl;
	    //theta = {3, -5, 1, 2};
	    //theta.print();
  	}



#if 1

    int nx = ns*nt;
    SpMat Qx(nx, nx);

    double t_Qx_factorise;
    RGF<T> *solver_Qx;
    solver_Qx = new RGF<T>(ns, nt, 0);

    double log_det_Qx;

    for(int c=0; c<10; c++){
        //theta = theta + Vect::Random(theta.size());
        std::cout << "\niter = " << c << ". Constructing precision matrix Qx. theta : " << theta.transpose() << std::endl;

        construct_Q_spat_temp(Qx, theta, c0, g1, g2, g3, M0, M1, M2);

        //SpMat epsId(nx,nx);
        //epsId.setIdentity();
        //epsId = 1e-4*epsId;
        //Qx = Qx + epsId;


        // only take lower triangular part of A
        SpMat Qx_lower = Qx.triangularView<Lower>(); 
        size_t nnz_Qx  = Qx_lower.nonZeros();

        //std::string Qx_lower_file = "Qst_lower_ns" + ns_s + "_nt" + nt_s + "_nb0_" + to_string(nx) + "_" + to_string(nx) + ".mtx";
        //Eigen::saveMarket(Qx_lower, Qx_lower_file);
        //exit(1);

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
        double flops_Qx_factorize = solver_Qx->factorize_noCopyHost(ia_Qx, ja_Qx, a_Qx, log_det_Qx);
        
        //double flops_Qx_factorize = solver_Qx->factorize(ia_Qx, ja_Qx, a_Qx);
        //log_det_Qx = solver_Qx->logDet(ia_Qx, ja_Qx, a_Qx);

        t_Qx_factorise = get_time(t_Qx_factorise);

        printf("logdet       : %f\n", log_det_Qx);
        printf("time chol(Qx): %lg\n",t_Qx_factorise);

        delete[] ia_Qx;
        delete[] ja_Qx;
        delete[] a_Qx;

    }

    delete solver_Qx;

#endif


#endif


#if 0


    size_t n = ns*nt + nb;

    SpMat Q(n,n);
    Vect rhs(n);
    double exp_theta = exp(theta[0]);
    rhs = exp_theta*Ax.transpose()*y;


    std::cout << "\nConstructing precision matrix Qxy. " << std::endl;
    //std::cout << "Setting Ax to zero." << std::endl;
    //Ax.makeCompressed();
    //Ax.setZero();

    RGF<T> *solver;
    solver = new RGF<T>(ns, nt, nb);

    //for(int c=0; c<1; c++){
        //theta = theta + Vect::Random(theta.size());
        //std::cout << "\niter = " << c << ". Constructing precision matrix Qxy. theta : " << theta.transpose() << std::endl;   

        construct_Q(Q, ns, nt, nb, theta, c0, g1, g2, g3, M0, M1, M2, Ax);
        //std::cout << "Q : \n" << Q.block(0,0,10,10) << std::endl;


        //SpMat epsId(n,n);
        //epsId.setIdentity();
        //epsId = 1e-4*epsId;

        //Q = Q + epsId;

    	// =========================================================================== //
    	std::cout << "Converting Eigen Matrices to CSR format. " << std::endl;

    	// only take lower triangular part of A
        SpMat Q_lower = Q.triangularView<Lower>(); 
        size_t nnz    = Q_lower.nonZeros();

        size_t nnz_invBlks = (ns+nb)*ns*nt + nb*nb; 

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

    	t_factorise = get_time(0.0);
    	//solver->solve_equation(GR);
    	double flops_factorize = solver->factorize(ia, ja, a);
    	t_factorise = get_time(t_factorise);

    	double log_det = solver->logDet(ia, ja, a);
    	printf("logdet: %f\n", log_det);

      	// assign b to correct format
      	for (int i = 0; i < n; i++){
    	    b[i] = rhs[i];
    	    //printf("%f\n", b[i]);
      	}

      	t_solve = get_time(0.0); 
        double flops_solve = solver->solve(ia, ja, a, x, b, 1);
      	t_solve = get_time(t_solve);
      	printf("flops solve:     %f\n", flops_solve);

    	printf("Residual norm.           : %e\n", solver->residualNorm(x, b));
    	printf("Residual norm normalized : %e\n", solver->residualNormNormalized(x, b));

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
   std::cout << "log Det Eigen : " << std::setprecision(10) << logDetEigen << std::endl;
   std::cout << "diff Log Dets : " << logDetEigen - log_det << std::endl;

   SpMat inv_Q = solverQ.solve(eye);
#endif


#if 0

    T *invDiag;
    T *invBlks;

    invDiag  = new T[n];
    invBlks  = new T[nnz_invBlks];

    double t_invDiag;
    double t_invBlks;

    t_invDiag = get_time(0.0);
    double flops_invDiag = solver->RGFdiag(ia, ja, a, invDiag);
    t_invDiag = get_time(t_invDiag);
    double log_detRGFdiag = solver->logDet(ia, ja, a);

    //printf("flops inv:      %f\n", flops_invDiag);

    
    t_invBlks = get_time(0.0);
    double flops_invBlks = solver->RGFinvBlks(ia, ja, a, invBlks);
    t_invBlks= get_time(t_invBlks);
    double log_detRGFinvBlks = solver->logDet(ia, ja, a);
    std::cout << "diff Log Dets : " << log_detRGFdiag - log_detRGFinvBlks << std::endl;



    printf("RGF factorise time: %lg\n",t_factorise);
    printf("RGF solve     time: %lg\n",t_solve);
    printf("RGF inv Diag  time: %lg\n",t_invDiag);
    printf("RGF inv Blks  time: %lg\n",t_invBlks);


#ifdef PRINT_MSG
    printf("mainEigen: array containing all neccessary inv blk entries: \n");
    for(int i=0; i<nnz_invBlks; i++){
            printf("%f ", invBlks[i]);
      }
      printf("\n");
#endif

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
    */
    
    SpMat QinvBlks(n,n);
    size_t nnz_full_invBlks = ns*ns*nt + 2*ns*nb*nt + nb*nb;
    double t_constrQinvBlks = -omp_get_wtime();
    construct_full_CSC_invBlks(ns, nt, nb, nnz_full_invBlks, invBlks, QinvBlks);
    t_constrQinvBlks += omp_get_wtime();
    printf("Assemble QinvBlks full time : %lg\n \n",t_constrQinvBlks);

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

    Vect invDiag_vec(n);
    // assign b to correct format
    for (int i = 0; i < n; i++){
        invDiag_vec[i] = invDiag[i];
        //printf("%f\n", b[i]);
    }

    Vect invDiagfBlks(n);
    invDiagfBlks = QinvBlks.diagonal();

   //cout << "Q:\n" << Q << endl;

    printf("norm(invDiag - invDiagfBlks)   : %f\n", (invDiag_vec - invDiagfBlks).norm());
    cout << "norm(invDiag - inv(Q))         : " << (invDiag_vec - inv_Q.diagonal()).norm() << std::endl;
    cout << "norm(diag(invQ)) : " << inv_Q.diagonal().norm() << std::endl;
    cout << "norm(invDiagfBlks) : " << invDiagfBlks.norm() << std::endl;
    cout << "norm(invDiag) : " << invDiag_vec.norm() << std::endl;


    if(n < 20){
        cout << "\ninvDiag BTA            : " << invDiag_vec.transpose() << std::endl;
        cout << "invDiag from blks BTA  : "   << invDiagfBlks.transpose() << std::endl;
        //cout << "Eigen diag(inv_Q)      : "   << inv_Q.diagonal().transpose() << endl;
    } else {
#ifdef PRINT_MSG
        cout << "\ninvDiag BTA[1:20]       : " << invDiag_vec.head(20).transpose() << std::endl;
        cout << "invDiag from blks BTA   : "   << invDiagfBlks.transpose() << std::endl;
        cout << "Eigen diag(inv_Q)[1:20] : " << inv_Q.diagonal().head(20).transpose() << endl;
        cout << "norm(invDiag - inv(Q))  : " << (invDiag - inv_Q.diagonal()).norm() << std::endl;
#endif   
    }


  delete[] invDiag;
  delete[] invBlks;

#endif

  
  // free memory
  delete solver;

  delete[] ia;
  delete[] ja;
  delete[] a;

  delete[] x;
  delete[] b;

  #endif

    
  return 0;


  }