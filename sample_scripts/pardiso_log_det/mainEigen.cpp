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

#include <armadillo>

// kaust path ... 
#include "/home/hpc/ihpc/ihpc060h/b_INLA/read_write_functions.cpp"

using Eigen::VectorXd;
using Eigen::MatrixXd;

typedef Eigen::VectorXd Vect;

#define PRINT_MSG

/* PARDISO prototype. */
extern "C" void pardisoinit (void   *, int    *,   int *, int *, double *, int *);
extern "C" void pardiso     (void   *, int    *,   int *, int *,    int *, int *, 
                  double *, int    *,    int *, int *,   int *, int *,
                     int *, double *, double *, int *, double *);
extern "C" void pardiso_chkmatrix  (int *, int *, double *, int *, int *, int *);
extern "C" void pardiso_chkvec     (int *, int *, double *, int *);
extern "C" void pardiso_printstats (int *, int *, double *, int *, int *, int *,
                           double *, int *);

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
        Qst = pow(exp_theta1,2)*(KroneckerProductSparse<SpMat, SpMat>(M0, q3s) + 2*exp_theta3 *KroneckerProductSparse<SpMat, SpMat>(M1, q2s) + pow(exp_theta3, 2)* KroneckerProductSparse<SpMat, SpMat>(M2, q1s));

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


/* ===================================================================== */

int main(int argc, char* argv[])
{

    if(argc != 1 + 5){
        std::cout << "wrong number of input parameters. " << std::endl;

        std::cerr << "INLA Call : ns nt nb no path/to/files" << std::endl;

        std::cerr << "[integer:ns]                number of spatial grid points " << std::endl;
        std::cerr << "[integer:nt]                number of temporal grid points " << std::endl;
        std::cerr << "[integer:nb]                number of fixed effects" << std::endl;
        std::cerr << "[integer:no]                number of data samples" << std::endl;

        std::cerr << "[string:base_path]          path to folder containing matrix files " << std::endl;
    
        exit(1);
    }

    std::cout << "reading in example. " << std::endl;

    int i; // iteration variable

    int ns = atoi(argv[1]);
    int nt = atoi(argv[2]);
    int nb = atoi(argv[3]);
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
	//theta << 1.391313, -5.913299,  1.076161,  3.642337;
        //theta << 5, -10, 2.5, 1;
        theta << 1.386294, -5.882541,  1.039721,  3.688879;
        //theta = {3, -5, 1, 2};
        //theta.print();
    }

    std::cout << "Constructing precision matrix Q. " << std::endl;

    int n = ns*nt + nb;
    SpMat Q(n,n);
    construct_Q(Q, ns, nt, nb, theta, c0, g1, g2, g3, M0, M1, M2, Ax);
    //std::cout << "Q : \n" << Q.block(0,0,10,10) << std::endl;

    Vect rhs(n);
    double exp_theta = exp(theta[0]);
    rhs = exp_theta*Ax.transpose()*y;

    // =========================================================================== // 
    // TEST MATRIX

    /*std::string test_mat_file      =  "/home/x_gaedkelb/b_INLA/data/test_matrix_11.dat";
    file_exists(test_mat_file);

    SpMat test_mat = read_sym_CSC(test_mat_file);

    std::cout << "test mat : \n" << test_mat << std::endl;*/


    // =========================================================================== //
        std::cout << "Converting Eigen Matrices to CSR format. " << std::endl;

    // only take lower triangular part of A
    //SpMat Q_lower = Q.triangularView<Lower>(); 
    //int nnz = Q_lower.nonZeros();

    SpMat Q_lower = Q.triangularView<Lower>(); 
    int nnz = Q_lower.nonZeros();
    n = Q_lower.rows();

    std::cout << "nnz " << nnz << std::endl;
    std::cout << "n " << n << std::endl;

    //std::cout << "test mat lower : \n" << Q_lower << std::endl;


    std::cout << "setting up matrices" << std::endl;

    int* ia; 
    int* ja;
    double* a; 

    // allocate memory
    ia = new int [n+1];
    ja = new int [nnz];
    a = new double [nnz];

    double* b = new double [n];
    // empty solution vector
    double* x = new double [n];

    for (int i = 0; i < n; ++i){
        b[i] = i;
    }

    Q_lower.makeCompressed();

    for (int i = 0; i < n+1; ++i){
        ia[i] = Q_lower.outerIndexPtr()[i]; 
         //std::cout << "ia[" << i << "] = " << ia[i] << std::endl;
    } 

    for (int i = 0; i < nnz; ++i){
        ja[i] = Q_lower.innerIndexPtr()[i];
            //std::cout << "ja[" << i << "] = " << ja[i] << std::endl;
    }  

    for (int i = 0; i < nnz; ++i){
        a[i] = Q_lower.valuePtr()[i];
            //std::cout << "a[" << i << "] = " << a[i] << std::endl;  
    }

    //exit(1);


    std::cout << "Setting up PARDISO parameters." << std::endl;
    // =========================================================================== //
    int nrhs = 1;          /* Number of right hand sides. */

  // must choose -2 for iterative solver
  int      mtype = -2;        /* Symmetric positive definite matrix */

  /* Internal solver memory pointer pt,                  */
  /* 32-bit: int pt[64]; 64-bit: long int pt[64]         */
  /* or void *pt[64] should be OK on both architectures  */ 
  void    *pt[64]; 

  /* Pardiso control parameters. */
  int      iparm[64];
  double   dparm[64];
  int      maxfct, mnum, phase, error, msglvl, solver;

  /* Number of processors. */
  int      num_procs;

  /* Auxiliary variables. */
  char    *var;
  int      k;

  double   ddum;              /* Double dummy */
  int      idum;              /* Integer dummy. */
   
/* -------------------------------------------------------------------- */
/* ..  Setup Pardiso control parameters.                                */
/* -------------------------------------------------------------------- */

    std::cout << "Calling PARDISO init." << std::endl;

    error  = 0;
    solver = 0;

    pardisoinit (pt,  &mtype, &solver, iparm, dparm, &error);

    std::cout << "error : " << error << std::endl;

    if (error != 0) 
    {
        if (error == -10 )
           printf("No license file found \n");
        if (error == -11 )
           printf("License is expired \n");
        if (error == -12 )
           printf("Wrong username or hostname \n");
        return 1; 
    }
    
    std::cout << "[PARDISO]: License check was successful ... " << std::endl;
    
    /* Numbers of processors, value of OMP_NUM_THREADS */
    var = getenv("OMP_NUM_THREADS");
    if(var != NULL)
        sscanf( var, "%d", &num_procs );
    else {
        printf("Set environment OMP_NUM_THREADS to 1");
        exit(1);
    }

    iparm[2]  = num_procs;

    maxfct = 1;         /* Maximum number of numerical factorizations.  */
    mnum   = 1;         /* Which factorization to use. */
    
    msglvl = 1;         /* Print statistical information  */
    error  = 0;         /* Initialize error flag */


/* -------------------------------------------------------------------- */
/* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
/*     notation.                                                        */
/* -------------------------------------------------------------------- */

    for (int i = 0; i < n+1; i++) {
        ia[i] += 1;
    }
    for (int i = 0; i < nnz; i++) {
        ja[i] += 1;
    }

    std::cout << "after 1 based conversion" << std::endl;


/* -------------------------------------------------------------------- */
/*  .. pardiso_chk_matrix(...)                                          */
/*     Checks the consistency of the given matrix.                      */
/*     Use this functionality only for debugging purposes               */
/* -------------------------------------------------------------------- */

    pardiso_chkmatrix  (&mtype, &n, a, ia, ja, &error);
    if (error != 0) {
        printf("\nERROR in consistency of matrix: %d", error);
        exit(1);
    }

/* -------------------------------------------------------------------- */
/* ..  pardiso_chkvec(...)                                              */
/*     Checks the given vectors for infinite and NaN values             */
/*     Input parameters (see PARDISO user manual for a description):    */
/*     Use this functionality only for debugging purposes               */
/* -------------------------------------------------------------------- */

    pardiso_chkvec (&n, &nrhs, b, &error);
    if (error != 0) {
        printf("\nERROR  in right hand side: %d", error);
        exit(1);
    }

/* -------------------------------------------------------------------- */
/* .. pardiso_printstats(...)                                           */
/*    prints information on the matrix to STDOUT.                       */
/*    Use this functionality only for debugging purposes                */
/* -------------------------------------------------------------------- */

    pardiso_printstats (&mtype, &n, a, ia, ja, &nrhs, b, &error);
    if (error != 0) {
        printf("\nERROR right hand side: %d", error);
        exit(1);
    }
 
/* -------------------------------------------------------------------- */
/* ..  Reordering and Symbolic Factorization.  This step also allocates */
/*     all memory that is necessary for the factorization.              */
/* -------------------------------------------------------------------- */

    std::cout << "After PARDISO checks." << std::endl;

    iparm[19-1] = -1; // in order to compute Gflops
    printf("\nGFlops factorisation : %i", iparm[19-1]);

    // start timer phase 1
    double timespent_p11 = -omp_get_wtime();

    phase = 11; 

    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
         &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error, dparm);
    
    if (error != 0) {
        printf("\nERROR during symbolic factorization: %d", error);
        exit(1);
    }

    // get time phase 1
    timespent_p11 += omp_get_wtime();
    // printf("\nTime spent on Phase 1 : %f", time_spent_p11);
   
/* -------------------------------------------------------------------- */
/* ..  Numerical factorization.                                         */
/* -------------------------------------------------------------------- */

    // start timer phase 2
    double timespent_p22 = -omp_get_wtime();

    phase = 22;
    iparm[32] = 1; /* compute determinant */

    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error,  dparm);
   
    if (error != 0) {
        printf("\nERROR during numerical factorization: %d", error);
        exit(2);
    }

    // get time phase 2
    timespent_p22 += omp_get_wtime();

    // printf("\nFactorization completed ...\n ");
    printf("\nFactorization completed .. \n");

    double log_det = dparm[32];
    printf("\nPardiso   log(det) = %f ", log_det);

    int gflops_fact = iparm[19-1];
    int mem_fact_solve = iparm[17-1];

    printf("\nGFlops factorisation : %i", iparm[19-1]);
    printf("\nMem fact + solve     : %i", mem_fact_solve);

/* -------------------------------------------------------------------- */    
/* ..  Back substitution and iterative refinement.                      */
/* -------------------------------------------------------------------- */    

    // start timer phase 3
    //double timespent_p33 = 0;
    double timespent_p33 = -omp_get_wtime();

    phase = 33;

    iparm[7] = 0;       /* Max numbers of iterative refinement steps. */
 
    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, b, x, &error,  dparm);
   
    if (error != 0) {
        printf("\nERROR during solution: %d", error);
        exit(3);
    }

    // get time phase 3
    timespent_p33 += omp_get_wtime(); 

    printf("\nSolve completed ... "); 

  /* -------------------------------------------------------------------- */    
  /* ... Inverse factorization.                                           */                                       
  /* -------------------------------------------------------------------- */  

  double timespent_sel_inv = 0; 

  if (solver == 0)
  {
    printf("\nCompute Diagonal Elements of the inverse of A ... \n");
    timespent_sel_inv = -omp_get_wtime();

    phase = -22;
    //iparm[35]  = 1; /*  no not overwrite internal factor L // crashes for larger matrices if uncommented */ 
    pardiso (pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &idum, &nrhs,
          iparm, &msglvl, b, x, &error,  dparm);
    
    // get time to compute selected inverse
    timespent_sel_inv += omp_get_wtime(); 


    /* print & save diagonal elements */
      std::string sel_inv_file_name = base_path + "/sel_inv_PARDISO_ns" + ns_s + "_nt" + nt_s + "_nb" + nb_s + "_no" + no_s + ".dat"; 
      std::ofstream sel_inv_file(sel_inv_file_name,    std::ios::out | std::ios::trunc);
      
    for (k = 0; k < n; k++){
      int j = ia[k]-1;
      sel_inv_file << a[j] << std::endl;

      //printf ("Diagonal element of A^{-1} = %d %d %32.24e\n", k, ja[j]-1, a[j]);
      //printf ("Diagonal element of A      = %d %d %32.24e\n", k, ja[j]-1, 1.0*Q_u(k,k));

      //printf ("Diagonal element of A^{-1}*A = %d %d %32.24e\n", k, ja[j]-1, a[j]*Q_u(k,k));

    }
  
    sel_inv_file.close();

  }  
  
/* -------------------------------------------------------------------- */    
/* ..  Convert matrix back to 0-based C-notation.                       */
/* -------------------------------------------------------------------- */ 
    for (int i = 0; i < n+1; i++) {
        ia[i] -= 1;
    }
    for (int i = 0; i < nnz; i++) {
        ja[i] -= 1;
    }


/* -------------------------------------------------------------------- */    
/* ..  Print statistics                                                 */
/* -------------------------------------------------------------------- */   

    printf("\nTime spent on phase 1 : %f s", timespent_p11);
    printf("\nTime spent on phase 2 : %f s", timespent_p22);
    printf("\nTime spent on phase 3 : %f s", timespent_p33);
    printf("\nTime spent on sel inv : %f s\n", timespent_sel_inv);


    // ================= //

    delete[] ja;
    delete[] ia;
    delete[] a;

    delete[] x;
    delete[] b;


  }
