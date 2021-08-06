#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <fstream>
#include <iostream>

#include <time.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>

//#define PRINT_MSG

#include "../read_write_functions.cpp"
#include "PardisoSolver.cpp"

using Eigen::VectorXd;
using Eigen::MatrixXd;

typedef Eigen::VectorXd Vector;
typedef Eigen::SparseMatrix<double> SpMat;

/* ===================================================================== */

/** spatial temporal model : SPDE discretisation. DEMF(1,2,1) model.*/
void construct_Q_spat_temp(Vector& theta, SpMat c0, SpMat g1, SpMat g2, SpMat g3, SpMat M0, SpMat M1, SpMat M2, SpMat* Qst){
    double exp_theta1 = exp(theta[1]);
    double exp_theta2 = exp(theta[2]);
    double exp_theta3 = exp(theta[3]);

    int nu = g1.rows()*M0.rows();

    // g^2 * fem$c0 + fem$g1
    SpMat q1s = pow(exp_theta2, 2) * c0 + g1;

     // g^4 * fem$c0 + 2 * g^2 * fem$g1 + fem$g2
    SpMat q2s = pow(exp_theta2, 4) * c0 + 2 * pow(exp_theta2,2) * g1 + g2;

    // g^6 * fem$c0 + 3 * g^4 * fem$g1 + 3 * g^2 * fem$g2 + fem$g3
    SpMat q3s = pow(exp_theta2, 6) * c0 + 3 * pow(exp_theta2,4) * g1 + 3 * pow(exp_theta2,2) * g2 + g3;

    #ifdef PRINT_MSG
        std::cout << "theta u : " << exp_theta1 << " " << exp_theta2 << " " << exp_theta3 << std::endl;

        /*std::cout << "pow(exp_theta1,2) : \n" << pow(exp_theta1,2) << std::endl;
        std::cout << "pow(exp_theta2,2) : \n" << pow(exp_theta2,2) << std::endl; 

        std::cout << "q1s : \n" << q1s.block(0,0,10,10) << std::endl;
        std::cout << "q2s : \n" << q2s.block(0,0,10,10) << std::endl;
        std::cout << "q3s : \n" << q3s.block(0,0,10,10) << std::endl;*/
    #endif

    // assemble overall precision matrix Q.st
    *Qst = pow(exp_theta1,2)*(KroneckerProductSparse<SpMat, SpMat>(M0, q3s) + 2*exp_theta3 *KroneckerProductSparse<SpMat, SpMat>(M1, q2s) + pow(exp_theta3, 2)* KroneckerProductSparse<SpMat, SpMat>(M2, q1s));

    //std::cout << "Qst : \n" << Qst->block(0,0,10,10) << std::endl;

}


void addFixedEffPrecision(int n, int nb, SpMat& Q, double val){

    /*int* outerIdxPtr;


    outerIdxPtr = new int[]

    Q.outerIndexPtr();
    int* innerIdxPtr = Q.innerIndexPtr();

    double* valuePtr = Q.valuePtr();*/




}

/** construct precision matrix. Calls spatial, spatial-temporal, etc. appropriately. */
void construct_Q(int ns, int nt, int nb, Vector& theta, SpMat c0, SpMat g1, SpMat g2, SpMat g3, SpMat M0, SpMat M1, SpMat M2, SpMat Ax, SpMat *Q){
    double exp_theta0 = exp(theta[0]);
    //double exp_theta = exp(3);

    int nu = ns*nt;
    int n = nu + nb;

    SpMat Q_b = 1e-5*Eigen::MatrixXd::Identity(nb, nb).sparseView(); 
    /*std::cout << "Q_b " << std::endl;
    std::cout << Eigen::MatrixXd(Q_b) << std::endl;*/
    //Q_b = 1e-5*Q_b.setIdentity();

    if(ns > 0){
        SpMat Qu(nu, nu);
        // TODO: find good way to assemble Qx

        if(nt > 1){
            construct_Q_spat_temp(theta, c0, g1, g2, g3, M0, M1, M2, &Qu);

        } else {    
            std::cout << "nt must be greater 1!" << std::endl;
            exit(1);
        }   

        int nnz_Qu = Qu.nonZeros();

        int* innerIndices;
        int* outerIndexPtr;
        double* values; 

        // allocate memory for overall matrix
        innerIndices  = new int [nnz_Qu+nb];
        outerIndexPtr = new int [nu+1+nb];
        values        = new double [nnz_Qu+nb];

        // copy csc structure of tridiagonal matrix into arrays
        memcpy(innerIndices,  Qu.innerIndexPtr(),  nnz_Qu*sizeof(int));
        memcpy(outerIndexPtr, Qu.outerIndexPtr(),  (nu+1)*sizeof(int));
        memcpy(values,        Qu.valuePtr(),       nnz_Qu*sizeof(double));

        // ------------- fill up precision matrix fixed effects ---------------- //

        double prec_val = 1e-5;

        int count = 0;
        for(int i = nnz_Qu; i<(nnz_Qu+nb); i++){
            innerIndices[i] = nu + count;
            values[i] = prec_val;
            count++;
        }

        for (int i = (nu+1); i < (nu+nb+1); i++){
            outerIndexPtr[i] = outerIndexPtr[i-1]+1;
        }

        int nnz    = nnz_Qu  + nb;
        SpMat Qx = Eigen::Map<Eigen::SparseMatrix<double> >(n,n,nnz,outerIndexPtr, // read-write
                               innerIndices,values);

        #ifdef PRINT_MSG
            //std::cout << "Qx : \n" << Qx.block(0,0,10,10) << std::endl;
            //std::cout << "Ax : \n" << Ax.block(0,0,10,10) << std::endl;

        #endif

        *Q =  Qx + exp_theta0 * Ax.transpose() * Ax;

        #ifdef PRINT_MSG
            std::cout << "exp(theta0) : \n" << exp_theta0 << std::endl;
            std::cout << "Qx dim : " << Qx.rows() << " " << Qx.cols() << std::endl;
        #endif

    } else {
            std::cout << "ns must be greater than zero!" << std::endl;
            exit(1);
    }

    /*std::cout << "Q -  exp(theta)*B'*B " << std::endl;
    std::cout << Eigen::MatrixXd(*Q) - exp_theta*B.transpose()*B << std::endl;*/

    #ifdef PRINT_MSG
        std::cout << "Q  dim : " << Q->rows() << " "  << Q->cols() << std::endl;
        std::cout << "Q : \n" << Q->block(0,0,10,10) << std::endl;
        std::cout << "theta : \n" << theta.transpose() << std::endl;

    #endif 

}
  
/** construct precision matrix. Calls spatial, spatial-temporal, etc. appropriately. */
void construct_Q_old(int ns, int nt, int nb, Vector& theta, SpMat c0, SpMat g1, SpMat g2, SpMat g3, SpMat M0, SpMat M1, SpMat M2, SpMat Ax, SpMat *Q){
    double exp_theta0 = exp(theta[0]);
    //double exp_theta = exp(3);

    int nu = ns*nt;
    int n = nu + nb;

    SpMat Q_b = 1e-5*Eigen::MatrixXd::Identity(nb, nb).sparseView(); 
    /*std::cout << "Q_b " << std::endl;
    std::cout << Eigen::MatrixXd(Q_b) << std::endl;*/
    //Q_b = 1e-5*Q_b.setIdentity();

    if(ns > 0){
        SpMat Qu(nu, nu);
        // TODO: find good way to assemble Qx

        if(nt > 1){
            construct_Q_spat_temp(theta, c0, g1, g2, g3, M0, M1, M2, &Qu);

        } else {    
            std::cout << "nt must be greater 1!" << std::endl;
            exit(1);
        }   

        //Qub0 <- sparseMatrix(i=NULL,j=NULL,dims=c(nb, ns))
        // construct Qx from Qs values, extend by zeros 
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

        *Q =  Qx + exp_theta0 * Ax.transpose() * Ax;

        #ifdef PRINT_MSG
            std::cout << "exp(theta0) : \n" << exp_theta0 << std::endl;
            std::cout << "Qx dim : " << Qx.rows() << " " << Qx.cols() << std::endl;
        #endif

        } else {
            std::cout << "ns must be greater than zero!" << std::endl;
            exit(1);
        }

    /*std::cout << "Q -  exp(theta)*B'*B " << std::endl;
    std::cout << Eigen::MatrixXd(*Q) - exp_theta*B.transpose()*B << std::endl;*/

    #ifdef PRINT_MSG
        std::cout << "Q  dim : " << Q->rows() << " "  << Q->cols() << std::endl;
        std::cout << "Q : \n" << Q->block(0,0,10,10) << std::endl;
        std::cout << "theta : \n" << theta.transpose() << std::endl;

    #endif 

}

/** Assemble right-handside. Could compute Ax^T*y once, and only multiply with appropriate exp_theta.*/
void construct_b(Vector& theta, SpMat Ax, Vector y, Vector *rhs){
    double exp_theta = exp(theta[0]);
    //double exp_theta = exp(3);

    *rhs = exp_theta*Ax.transpose()*y;

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

    size_t ns = atoi(argv[1]);
    size_t nt = atoi(argv[2]);
    size_t nb = atoi(argv[3]);
    size_t no = atoi(argv[4]);

    // set nt = 1 if ns > 0 & nt = 0
    if(ns > 0 && nt == 0){
        nt = 1;
    } 

    // also save as string
    std::string ns_s = std::to_string(ns);
    std::string nt_s = std::to_string(nt);
    std::string nb_s = std::to_string(nb);
    std::string no_s = std::to_string(no); 
    std::string n_s  = std::to_string(ns*nt + nb);

    std::string base_path = argv[5];

    // dimension hyperparamter initialised to 1
    int dim_th = 1;

    /* ---------------- read in matrices ---------------- */

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
    Vector y;

    if(ns == 0 && nt == 0){
        // read in design matrix 
        // files containing B
        std::string B_file        =  base_path + "/B_" + no_s + "_" + nb_s + ".dat";
        file_exists(B_file); 

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

        Ax = readCSC(Ax_file);

        /*std::cout << "g1 : \n" << g1.block(0,0,10,10) << std::endl;
        std::cout << "g2 : \n" << g2.block(0,0,10,10) << std::endl;

        std::cout << "Ax : \n" << Ax.block(0,0,10,10) << std::endl;*/

    } else if(ns > 0 && nt > 1) {

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

    } else {
        std::cout << "invalid parameters : ns nt !!" << std::endl;
        exit(1);
    }


    // data y
    std::string y_file        =  base_path + "/y_" + no_s + "_1" + ".dat";
    file_exists(y_file);
    y = read_matrix(y_file, no, 1);


    /* ----------------------- initialise random theta -------------------------------- */

    Vector theta(dim_th);
    Vector theta_original(dim_th);

    int n;

    // initialise theta
    if(ns == 0 && nt == 0){
        n = nb;
        // Initial guess
        theta[0] = 3;
        std::cout << "initial theta : "  << theta.transpose() << std::endl;    

    } else if(ns > 0 && nt == 1){
        n = ns + nb;
        //theta << 1, -1, 1;
        theta << 1, -2, 2;
        std::cout << "initial theta : "  << theta.transpose() << std::endl;    

    } else {
        n = ns*nt + nb;

        // =========== synthetic data set =============== //
        /*std::cout << "using SYNTHETIC DATASET" << std::endl;        
        theta_original << 1.4, -5.9,  1,  3.7; 
        std::cout << "theta original     : " << std::right << std::fixed << theta_original.transpose() << std::endl;
        theta << 1.4, -5.9,  1,  3.7; 
        //theta << 1, -3, 1, 3;
        //theta << 0.5, -1, 2, 2;
        std::cout << "initial theta      : "  << std::right << std::fixed << theta.transpose() << std::endl;*/

        // =========== temperature data set =============== //
        std::cout << "using TEMPERATURE DATASET" << std::endl;        
        theta_original << 5, -10, 2.5, 1;
        std::cout << "theta original     : " << std::right << std::fixed << theta_original.transpose() << std::endl;
        theta << 5, -10, 2.5, 1;
        std::cout << "initial theta      : "  << std::right << std::fixed << theta.transpose() << std::endl;


    }

    /* ====================================================================== */


    Eigen::MatrixXd M_dense(4,4);

    M_dense << 1,0,0,3.5,
               4,2,1,0,
               3,0,1,0,
               0,1,0,1;


    std::cout << "M_dense : \n" << M_dense << std::endl; 

    SpMat M = M_dense.sparseView();

    double val = 0.5;
    //int nb  = 2;

    std::cout << "M : \n" << M << std::endl; 

    int rows_M = M.rows();
    int cols_M = M.cols();
    int nnz_M    = M.nonZeros();

    int* innerIndices;
    int* outerIndexPtr;
    double* values; 

    // allocate memory
    innerIndices  = new int [nnz_M+nb];
    outerIndexPtr = new int [cols_M+1+nb];
    values        = new double [nnz_M+nb];

    // copy csc structure of tridiagonal matrix into arrays
    memcpy(innerIndices, M.innerIndexPtr(), nnz_M*sizeof(int));
    memcpy(outerIndexPtr, M.outerIndexPtr(), (cols_M+1)*sizeof(int));
    memcpy(values, M.valuePtr(), nnz_M*sizeof(double));

    #if 0

    std::cout << "inner index ptr" << std::endl;
    for(int i=0; i<nnz_M; i++){
        std::cout << M.innerIndexPtr()[i] << std::endl;
    }

    std::cout << "new inner index ptr" << std::endl;
    for(int i=0; i<nnz_M; i++){
        std::cout << innerIndices[i] << std::endl;
    }


    std::cout << "sizeof(int)               = " << sizeof(int) << std::endl;

    std::cout << "outer index ptr" << std::endl;
    for(int i=0; i<cols_M+1; i++){
        std::cout << M.outerIndexPtr()[i] << std::endl;
    }

    std::cout << "new outer index ptr" << std::endl;
    for(int i=0; i<cols_M+1; i++){
        std::cout << outerIndexPtr[i] << std::endl;
    }



    // allocate memory
    /*int innerIndices[nnz];
    std::cout << "inner index ptr" << std::endl;
    int outerIndexPtr[n_cols+1];
    double val[nnz];
    std::cout << "inner index ptr" << std::endl;
    double values[nnz];*/

    std::cout << "inner index ptr" << std::endl;
    for (int i = 0; i < nnz_M; i++){
        innerIndices[i] = M.innerIndexPtr()[i];
        values[i] = M.valuePtr()[i];
        std::cout << innerIndices[i] << std::endl;
    }


    std::cout << "outerIndexPtr : " << std::endl;
    for (int i = 0; i < cols_M+1; i++){
        outerIndexPtr[i] = M.outerIndexPtr()[i];
        std::cout << outerIndexPtr[i] << std::endl;
    }


    #endif

    // ------------- fill up precision matrix fixed effects ---------------- //

    std::cout << "added values : " << std::endl;

    int count = 0;
    for(int i = nnz_M; i<(nnz_M+nb); i++){
        innerIndices[i] = cols_M + count;
        values[i] = val;
        count++;
    }

    std::cout << "outerIndexPtr : " << std::endl;
    for (int i = (cols_M+1); i < (cols_M+nb+1); i++){
        outerIndexPtr[i] = outerIndexPtr[i-1]+1;
        std::cout << outerIndexPtr[i] << std::endl;

    }

    int n_rows = rows_M + nb;
    int n_cols = cols_M + nb; 
    int nnz    = nnz_M  + nb;

    std::cout << "inner index ptr" << std::endl;
    for(int i=0; i<nnz; i++){
        std::cout << innerIndices[i] << std::endl;
    }

   std::cout << "outer index ptr" << std::endl;
    for(int i=0; i<n_cols+1; i++){
        std::cout << outerIndexPtr[i] << std::endl;
    }

    std::cout << "values" << std::endl;
    for(int i=0; i<nnz; i++){
        std::cout << values[i] << std::endl;
    }



    // 
    SpMat A = Eigen::Map<Eigen::SparseMatrix<double> >(n_rows,n_cols,nnz,outerIndexPtr, // read-write
                               innerIndices,values);


    std::cout << "A = \n" << MatrixXd(A) << std::endl;


    int* a;
    int* b;
    //int* c;

    a = new int [5];
    b = new int [5];
    //c = new int [10];

    for(int i=0; i < 5; i++){
        a[i] = i+1;
        b[i] = i+6;
        //cout << a[i] << endl;
        //cout << b[i] << endl;

    }

    /*int a[] = {1,2,3,4,5};
    int b[] = {6,7,8,9,10};
    int c[10];

    memcpy( c[0], a, sizeof(int) * sizeof(a) );
    memcpy( c[5], b, sizeof(int) * sizeof(b) );*/


    int * c = new int[5 + 5];
    std::copy(a, a + 5, c);
    std::copy(b, b + 5, c + 5);



    /*for(int i=0; i<10; i++)
    {
       cout << c[i] << endl;  
    }*/

    // construct precision matrix Q
    #if 0

    double construct_Q_time = -omp_get_wtime();
    SpMat Q(n,n);
    construct_Q(ns, nt, nb, theta, c0, g1, g2, g3, M0, M1, M2, Ax, &Q);
    //std::cout << "Q : \n" << Q.block(0,0,10,10) << std::endl;
    construct_Q_time += omp_get_wtime();

    std::cout << "construct_Q_time = " << construct_Q_time << std::endl;

    double construct_Qst_time = -omp_get_wtime();
    SpMat Qst(ns*nt, ns*nt);
    std::cout << "dim Qst : " << ns*nt << ", actual " << Qst.block(0,0,ns*nt, ns*nt).rows() << std::endl;
    construct_Q_spat_temp(theta, c0, g1, g2, g3, M0, M1, M2, &Qst);
    construct_Qst_time += omp_get_wtime();

    std::cout << "construct_Qst_time = " << construct_Qst_time << std::endl;

    #endif


    #if 0 
    //exit(1);

    double overall_time = -omp_get_wtime();

    int num_par = 4;

    int threads = std::atoi(getenv("OMP_NUM_THREADS"));
    std::cout << "OMP NUM THREADS = " << threads << std::endl;

    PardisoSolver* solverQ[num_par];
    PardisoSolver* solverQst[num_par];

    // create as many Solvers as there are threads
    for(int i=0; i<num_par; i++){
        solverQ[i] = new PardisoSolver();
        solverQst[i] = new PardisoSolver();
    }

    for(int bfgs_iter = 0; bfgs_iter < 1; bfgs_iter++)
    {
        std::cout << "=============== bfgs iter = " << bfgs_iter << " ================= " << std::endl;

        #pragma omp parallel for   
        for(int i=0; i<num_par; i++){

            std::cout << "==== i = " << i << std::endl;

            int tid = omp_get_thread_num();
            std::cout << "tid = " << tid << std::endl;

            // ------------------ initialise & construct matrices & rhs --------------------- //
            double construct_Q_time = -omp_get_wtime();
            SpMat Q(n,n);
            construct_Q(ns, nt, nb, theta, c0, g1, g2, g3, M0, M1, M2, Ax, &Q);
            //std::cout << "Q : \n" << Q.block(0,0,10,10) << std::endl;
            construct_Q_time += omp_get_wtime();


            Vector rhs(n);
            construct_b(theta, Ax, y, &rhs);
            //std::cout << "b : \n" << rhs.head(10) << std::endl;

            double construct_Qst_time = -omp_get_wtime();
            SpMat Qst(ns*nt, ns*nt);
            construct_Q_spat_temp(theta, c0, g1, g2, g3, M0, M1, M2, &Qst);
            construct_Qst_time += omp_get_wtime();

            std::cout << "construct_Q_time = " << construct_Q_time << std::endl;
            std::cout << "construct_Qst_time = " << construct_Qst_time << std::endl;


            SpMat IdMat(n,n); IdMat.setIdentity();
            //std::cout << "Q : \n" << Q.block(0,0,10,10) << std::endl;
            SpMat W = Q + i*IdMat;
            //std::cout << "W : \n" << W.block(0,0,10,10) << std::endl;

            Vector x(n);
            Vector inv_diag(n);

            double log_det_Q;
            double log_det_Qst;

            // --------------------- call solvers --------------------- //

            // factorise matrix
            solverQst[i]->factorize(Qst, log_det_Qst); 
            std::cout << "log det factorise Qst      : " << log_det_Qst << std::endl;

            // factorise & solve
            //W = Q + i*IdMat;
            solverQ[i]->factorize_solve(W, rhs, x, log_det_Q);
            std::cout << "log det factorise Q        : " << log_det_Q << std::endl;
     
            if(i == 0 && bfgs_iter == 1){
                string sol_x_file_name = base_path +"/pardiso_sol_x_ns"+to_string(ns)+"_nt"+to_string(nt)+"_nb"+ nb_s + "_no" + no_s +".dat";
                write_vector(sol_x_file_name, x, n); 
            }

            // selected inversion
            solverQ[i]->selected_inversion(Q, inv_diag);

            // write to file
            if(i == 0 && bfgs_iter == 2){
                string sel_inv_file_name = base_path +"/pardiso_sel_inv_ns"+to_string(ns)+"_nt"+to_string(nt)+"_nb"+ nb_s + "_no" + no_s +".dat";
                write_vector(sel_inv_file_name, inv_diag, n);
            }

        }
    } // end bfgs iter

    overall_time += omp_get_wtime();
    std::cout << "run time : " << overall_time << std::endl;

    for(int i=0; i<num_par; i++){
        delete solverQ[i];
        delete solverQst[i];
    

    }

    #endif




    return 0;
}

