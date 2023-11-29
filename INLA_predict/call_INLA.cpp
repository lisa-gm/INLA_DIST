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

// enable BTA solver or not
#define BTA_SOLVER

#ifdef BTA_SOLVER
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

// y_predict_var expects: diag(A*Qinv*A^T) + 1/precision_obs -> need to do exp() of theta[0]
// returns: 1st column: DS, 2nd column: CRPS, 3rd column: SCRPS
MatrixXd scoreFunction(Vect& y_observed, Vect& y_predict_mean, Vect& y_predict_var){

    int dimy = y_observed.size();
    MatrixXd M(dimy, 3);
    M.setZero();

    // we can leave NA's -> it's ok, they will be taken out during post processing

    Vect y_diff = y_observed - y_predict_mean;
    // need to work with arrays
    // ds = y_diff.^2 ./ y_predict_var + log(v)
    M.col(0) = y_diff.array() * y_diff.array() / y_predict_var.array() + y_predict_var.array().log();

    // crps.g = y_predict_var / sqrt(pi) - 2*y_predict_var*dnorm(y_diff/s) + y_diff*(1-2*pnorm(y_diff/s))
    // dnorm -> evaluate standard normal distr. with zero mean, sd = 1, at x = y_diff/s
    Vect y_diffbyVar = y_diff.array() / y_predict_var.array();
    // dnorm: 1/sqrt(2*M_PI) exp(x^2 / 2)
    Vect dnorm_ydiffVar(dimy);
    dnorm_ydiffVar = 1 / sqrt(2*M_PI) * exp(-y_diffbyVar.array()*y_diffbyVar.array()/2);
    // pnorm: 1/2 (1 + erf(x / sqrt(2))) -> erf(x)
    Vect pnorm_yDiffvar(dimy);
    for(int i=0; i<dimy; i++){
      pnorm_yDiffvar[i] = 0.5 * (1 + erf(y_diffbyVar[i]/sqrt(2)));
    }
    printf("y/s = %f, dnorm = %f, pnorm = %f\n", y_diffbyVar[0], dnorm_ydiffVar[0], pnorm_yDiffvar[0]);

    M.col(1) =1/sqrt(M_PI) * y_predict_var.array() 
               - 2*y_predict_var.array() * dnorm_ydiffVar.array()
               + y_diff.array()*(ArrayXd::Ones(dimy) - 2*pnorm_yDiffvar.array());

    // scrps.g 
    // md = y_obs - y_pred
    //  -0.5 * log(2 * s/sqrt(pi)) - sqrt(pi) * (s * dnorm(md/s) -  md/2 + md * pnorm(md/s))/s
    M.col(2) = - 0.5*(2*y_predict_var.array()/sqrt(M_PI)).log()
               - sqrt(M_PI)/y_predict_var.array()*(y_predict_var.array()*dnorm_ydiffVar.array() - 0.5*y_diff.array() + y_diff.array() * pnorm_yDiffvar.array());

    return M;
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
	threads_level1 = omp_get_max_threads();
	//threads_level2 = omp_get_max_threads();
    	threads_level2 = 1;
    }

    // overwrite in case BTA is used
    int noGPUs;
   
    if(MPI_rank == 0){
        printf("\n============== PARALLELISM & NUMERICAL SOLVERS ==============\n");
        printf("total no MPI ranks  : %d\n", MPI_size);
        printf("OMP threads level 1 : %d\n", threads_level1);
        //printf("OMP threads level 2 : %d\n", threads_level2);
	printf("OMP threads level 2 FIXED TO 1!!\n");
#ifdef BTA_SOLVER
	cudaGetDeviceCount(&noGPUs);
	printf("available GPUs      : %d\n", noGPUs);
#else
	printf("BTA dummy version\n");
    noGPUs = 0;
#endif
    }  
    
    
    if(argc != 1 + 9 && MPI_rank == 0){
        std::cout << "wrong number of input parameters. " << std::endl;

        std::cerr << "INLA Call : ns nss nt_fit nt_pred nt_total no_per_ts nb path/to/files solver_type" << std::endl;
        //std::cerr << "INLA Call : ns nss nt_fit nt_pred nt_total no nb path/to/files solver_type" << std::endl;

        std::cerr << "[integer:ns]                number of spatial grid points " << std::endl;
        std::cerr << "[integer:nss]               number of spatial grid points add. spatial field" << std::endl;

        std::cerr << "[integer:nt_fit]            number of days used for fitting" << std::endl;
        std::cerr << "[integer:nt_pred]           number of days predicted" << std::endl;
        std::cerr << "[integer:nt_total]          number of days for which we have data" << std::endl;

        std::cerr << "[integer:no_per_ts]         number of data samples per ts (includes NA)" << std::endl;
        //std::cerr << "[integer:no]                total number of data samples per ts (includes NA)" << std::endl;
        std::cerr << "[integer:nb]                number of fixed effects" << std::endl;

        std::cerr << "[string:base_path]          path to folder containing matrix files " << std::endl;
        std::cerr << "[string:solver_type]        BTA, PARDISO or Eigen" << std::endl;

        exit(1);
    }

#ifdef PRINT_MSG
    if(MPI_rank == 0){
        std::cout << "reading in example. " << std::endl;
    }
#endif

    size_t ns  = atoi(argv[1]);
    size_t nss = atoi(argv[2]);

    size_t nt_fit    = atoi(argv[3]);
    size_t nt_pred   = atoi(argv[4]);
    size_t nt        = nt_fit + nt_pred; // internal model size
    size_t nt_total  = atoi(argv[5]);
    
    size_t no_per_ts = atoi(argv[6]);
    size_t no        = nt*no_per_ts;
    //size_t no        = atoi(argv[6]);
    size_t nb        = atoi(argv[7]);

    // set nt = 1 if ns > 0 & nt = 0
    if(ns > 0 && nt == 0){
        nt = 1;
    } 

    size_t n = ns*(nt_fit + nt_pred) + nss + nb;

    // also save as string
    std::string ns_s        = std::to_string(ns);
    std::string nt_s        = std::to_string(nt);    


    std::string nt_fit_s    = std::to_string(nt_fit);
    std::string nt_pred_s   = std::to_string(nt_pred);
    std::string nt_total_s  = std::to_string(nt_total);

    std::string no_per_ts_s = std::to_string(no_per_ts); 
    //std::string no_s        = std::to_string(no);
    std::string nb_s        = std::to_string(nb);
    std::string n_s         = std::to_string(n);

    std::string base_path   = argv[8];    
    std::string solver_type = argv[9];

    // check if solver type is neither PARDISO nor BTA :
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

    /* ---------------- read in matrices ---------------- */

    // dimension hyperparamter vector
    int dim_th;
    int dim_spatial_domain;
    string manifold = ""; // if empty or unknown -> R^d, for now add only sphere

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
    //num_constr = 1;
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

        dim_th = 2;

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

        if(nss == 0){
            dim_th = 4;
        } else {
            dim_th = 6;
        }

        if(MPI_rank == 0){
            printf("spatial-temporal model");
            if(nss > 0){
                printf(" with add. spatial field");
            }
            printf(".\n");
        }

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
        std::string Ax_file     =  base_path + "/Ax_all_" + to_string(no) + "_" + to_string(n) + ".dat";
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
        //size_t no_all = nt_total*no_per_ts;
        size_t no_all = no;
        std::string y_file        =  base_path + "/y_all_" + to_string(no_all) + "_1" + ".dat";
        file_exists(y_file);
        // at this point no is set ... 
        // not a pretty solution. 
        y_all = read_matrix(y_file, no_all, 1);  

        std::string y_ind_file        =  base_path + "/y_indicator_" + to_string(no_all) + "_1" + ".dat";
        file_exists(y_ind_file);
        // at this point no is set ... 
        // not a pretty solution. 
        y_ind = read_matrix(y_ind_file, no_all, 1); 

        /*
        std::string y_times_file        =  base_path + "/y_times_" + to_string(no_all) + "_1" + ".dat";
        file_exists(y_times_file);
        // at this point no is set ... 
        // not a pretty solution. 
        y_times = read_matrix(y_times_file, no_all, 1); 
        */

        no = y_ind.sum(); 

        if(MPI_rank == 0){
            std::cout << "total length y : " << no_all << ", total missing : " << no_all - no << std::endl;
            //std::cout << "total number of missing observations: " << no_all - no << std::endl;
            std::cout << "sum(y_ind) = " << y_ind.sum() << std::endl;
            std::cout << "y(1:10) = " << y_all.head(10).transpose() << std::endl;
        }

#else
        // check projection matrix for A.st
        std::string Ax_file     =  base_path + "/Ax_" + to_string(no) + "_" + n_s + ".dat";
        file_exists(Ax_file);

        Ax = readCSC(Ax_file);
        no = Ax.rows();

    // data y
        std::string y_file        =  base_path + "/y_" + to_string(no) + "_1" + ".dat";
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

        // constant in conversion between parametrisations changes dep. on spatial dim
        // assuming sphere -> assuming R^3
        dim_spatial_domain = 2;

        // define if on the sphere or plane
        manifold = "sphere";

        // =========== synthetic data set =============== //
        if(MPI_rank == 0){ 
            std::cout << "using SYNTHETIC DATASET" << std::endl; 
            if(manifold == "sphere"){
                std::cout << "spatial domain: " << manifold << std::endl;
            }
            else if(manifold.length() > 0){
                std::cout << "spatial domain: " << manifold << ", only SPHERE supported!" << std::endl;
                exit(1);
            }
        }     

        if(nss == 0){
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
        } else {
            // order prec obs, lgamS for st , lgamT for st, lgamE for st, lgamE for s, lgamS for s
            //theta_original     << 1.386294,     -4.469624,      0.6342557,    1.673976, -4.607818, 2.243694;
            theta_original << 0,0,0,0,0,0;
            //theta_prior_param     << 1.386294,     -4.469624,      0.6342557,    1.673976, -4.607818, 2.243694;
            // order: prec obs, range s for st, range t for st, prec sigma for st, range s for s, prec sigma for s
            //theta_prior_param  << -log(0.01)/5, -log(0.01)*0.1, -log(0.01)*1, -log(0.01)/1, -log(0.01)*(3000.0/6371.0), -log(0.01)/5;
            theta_prior_param  << -log(0.01)/5, -log(0.01)*pow(0.1, 0.5*dim_spatial_domain), -log(0.01)*pow(1, 0.5), -log(0.01)/3,-log(0.01)*pow(3000.0/6371.0, 0.5*dim_spatial_domain), -log(0.01)/5;
            if(MPI_rank == 0){
                std::cout << "theta prior param : " << theta_prior_param.transpose() << std::endl;
            }
            // same order as above
            //theta_param        << 1.4228949,     0.4164521,      1.0990791,    1.4407530,  -1.1989102, 1.1071601;
            theta_param <<  -2.137, -1.350, 2.713, 1.337, -0.088, 2.476;
            //theta_param << 4.000, 1.000, 1.000, 0.000, -0.852, 0.609; // temperature dataset!
        }

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
        dim_spatial_domain = 2;

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
    // CORRECTED PRIOR
	theta_prior_param[0] = -log(0.01)/5; 	      //prior.sigma obs : 5, 0.01
	//theta_prior_param[1] = -log(0.5)*1000;        //prior.rs=c(1000, 0.5), ## P(range_s < 1000) = 0.5
	theta_prior_param[1] = -log(0.01)*pow(500, 0.5*dim_spatial_domain);        
    //theta_prior_param[2] = -log(0.5)*20;	      //prior.rt=c(20, 0.5), ## P(range_t < 20) = 0.5
	theta_prior_param[2] = -log(0.01)*pow(1, 0.5);
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
    param.epsilon = 1e-2;
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

#ifdef PREDICT
    // extract appropriate columns from y
    y = y_all(Eigen::seq(nt_init_fit*no_per_ts, (nt_last_pred+1)*no_per_ts-1));
    Vect y_ind_sub = y_ind(Eigen::seq(nt_init_fit*no_per_ts, (nt_last_pred+1)*no_per_ts-1));
    //y = y_all;
    //Vect y_ind_sub = y_ind;

    no = y_ind_sub.sum();
    if(MPI_rank == 0){
       printf("first index: %ld, last index: %ld\n", nt_init_fit*no_per_ts, (nt_last_pred+1)*no_per_ts-1);
        printf("length(y) = %ld, length(y_ind_sub) = %ld, no(w/out NA) = %ld, rows(Ax) = %ld\n", y.size(), y_ind_sub.size(), no, Ax.rows());
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

#endif



    // ============================ set up Posterior of theta ======================== //

    //std::optional<PostTheta> fun;
    PostTheta* fun;

    if(ns == 0 && nss == 0){
        // fun.emplace(nb, no, B, y);
        if(MPI_rank == 0){
            std::cout << "Call constructor for regression model." << std::endl;
        }
        fun = new PostTheta(ns, nt, nb, no, B, y, theta_prior_param, solver_type, constr, Dxy, validate, w);
    } else if(ns > 0 && nt == 1 && nss == 0) {
        if(MPI_rank == 0){
            std::cout << "\ncall spatial constructor." << std::endl;
        }
        // PostTheta fun(nb, no, B, y);
        fun = new PostTheta(ns, nt, nb, no, Ax, y, c0, g1, g2, theta_prior_param, solver_type, dim_spatial_domain, manifold, constr, Dx, Dxy, validate, w);
    } else if(ns > 0 && nt > 1 && nss == 0){
        if(MPI_rank == 0){
            std::cout << "\ncall spatial-temporal constructor." << std::endl;
        }
        fun = new PostTheta(ns, nt, nb, no, Ax, y, c0, g1, g2, g3, M0, M1, M2, theta_prior_param, solver_type, dim_spatial_domain, manifold, constr, Dx, Dxy, validate, w);
    } else if(ns > 0 && nt > 1 && nss > 0){
         if(MPI_rank == 0){
            std::cout << "\ncall spatial-temporal constructor with add. spatial field." << std::endl;
        }       
        fun = new PostTheta(ns, nt, nss, nb, no, Ax, y, c0, g1, g2, g3, M0, M1, M2, theta_prior_param, solver_type, dim_spatial_domain, manifold, constr, Dx, Dxy, validate, w);
    } else {
        printf("invalid combination of parameters!\n");
        printf("ns = %ld, nt = %ld, nss = %ld\n", ns, nt, nss);
        exit(1);
    }


    if(MPI_rank == 0)
        printf("\n======================= HYPERPARAMETERS =====================\n");

    if(MPI_rank == 0){
        std::cout << "theta original     : " << std::right << std::fixed << theta_original.transpose() << std::endl;
        std::cout << "theta prior param  : " << theta_prior_param.transpose() << std::endl;
    }

    // convert from interpretable parametrisation to internal one
    if(dim_th >= 4){
	   theta[0] = theta_param[0];
        fun->convert_interpret2theta_spatTemp(theta_param[1], theta_param[2], theta_param[3], theta[1], theta[2], theta[3]);
        if(dim_th == 6){
            fun->convert_interpret2theta_spat(theta_param[4], theta_param[5], theta[4], theta[5]);
        }

        if(MPI_rank == 0){
            Vect theta_interpret_initial(dim_th);
            theta_interpret_initial[0] = theta[0];
            fun->convert_theta2interpret_spatTemp(theta[1], theta[2], theta[3], theta_interpret_initial[1], theta_interpret_initial[2], theta_interpret_initial[3]);
            if(dim_th == 6){
                fun->convert_theta2interpret_spat(theta[4], theta[5], theta_interpret_initial[4], theta_interpret_initial[5]);
            }
            std::cout << "theta interpret. param.         : " << theta_param.transpose() << std::endl;
	        std::cout << "initial theta                   : "  << std::right << std::fixed << theta.transpose() << std::endl;
            std::cout << "initial theta interpret. param. : " << theta_interpret_initial.transpose() << std::endl;
        }
    }

#ifdef WRITE_RESULTS
   string results_folder = base_path + "/results_param_INLAmode";
   if(MPI_rank == 0){
    	create_folder(results_folder);
   }
#endif




#if 1 
    if(MPI_rank == 0){
        printf("Constructing Q matrices for testing.\n");
        //theta_param <<  -2.14715194, -1.35357150, 2.70798981, 1.36344400, -0.08755905, 2.47428186;
        //theta = -1.350436 2.712842 1.331847 gamma = -8.755903 2.390157 6.800008
        theta_param << -2.136686, -1.350436, 2.712842, 1.331847, -0.088195, 2.405772;

        if(dim_th >= 4){
            theta[0] = theta_param[0];
            fun->convert_interpret2theta_spatTemp(theta_param[1], theta_param[2], theta_param[3], theta[1], theta[2], theta[3]);
            if(dim_th == 6){
                fun->convert_interpret2theta_spat(theta_param[4], theta_param[5], theta[4], theta[5]);
            }
        } 

        std::cout << "theta param: " << theta_param.transpose() << std::endl;
        std::cout << "theta      : " << theta.transpose()       << std::endl; 

        SpMat Q(n,n);
        fun->construct_Q(theta, Q);
        std::cout << "Q(1:10,1:10):\n" << Q.block(0,0,20,20) << std::endl;
        std::cout << "Q(-20:end,-20:end):\n" << Q.block(n-20,n-20,20,20) << std::endl;
        std::string Q_file = base_path + "/Qxy_" + n_s + "_" + n_s + ".dat";
        write_sym_CSC_matrix(Q_file, Q);  

        SpMat Qprior(n,n);
        fun->construct_Qprior(theta, Qprior);
        std::cout << "Qprior(1:10,1:10):\n" << Qprior.block(0,0,20,20) << std::endl;
        std::cout << "Qprior(-20:end,-20:end):\n" << Qprior.block(n-20,n-20,20,20) << std::endl;
        std::string Qprior_file = base_path + "/Qprior_" + n_s + "_" + n_s + ".dat";
        write_sym_CSC_matrix(Qprior_file, Qprior);  

        SpMat Q_all = Qprior + exp(theta[0])*Ax_all.transpose()*Ax_all;
        std::string Q_all_file = base_path + "/Qall_" + n_s + "_" + n_s + ".dat";
        std::cout << "Q_all(1:10,1:10):\n" << Q_all.block(0,0,20,20) << std::endl;
        std::cout << "Q_all(-20:end,-20:end):\n" << Q_all.block(n-20,n-20,20,20) << std::endl;
        write_sym_CSC_matrix(Q_all_file, Q_all);  

        Vect mu(n);
        fun->eval_post_theta(theta, mu);
    }


    exit(1);

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
    fun->convert_interpret2theta_spatTemp(theta_param[1], theta_param[2], theta_param[3], theta[1], theta[2], theta[3]);
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


#if 0

    if(MPI_rank == 0){

    double estLogDetQst;
    int nt_approx;
    SpMat Qst_approx;

    //theta << 1, -3, 2, 4;
    //theta = theta_original;

    // construct Qst_approx 
    nt_approx = 5; //floor(nt/10.0); //nt-2;
    fun->eval_log_prior_lat_approx(theta, nt_approx, estLogDetQst);
    std::cout << "\nnt : " << nt_approx << ", estLogDetQst   : " << estLogDetQst << std::endl;

    /*
    nt_approx = 10; //floor(nt/10.0); //nt-2;
    Qst_approx.resize(nt_approx*ns, nt_approx*ns);
    fun->construct_Q_spat_temp_approx(theta, nt_approx, Qst_approx, estLogDetQst);
    std::cout << "nt : " << nt_approx << ", estLogDetQst   : " << estLogDetQst << std::endl;

    nt_approx = 20; //floor(nt/10.0); //nt-2;
    Qst_approx.resize(nt_approx*ns, nt_approx*ns);
    fun->construct_Q_spat_temp_approx(theta, nt_approx, Qst_approx, estLogDetQst);
    std::cout << "nt : " << nt_approx << ", estLogDetQst   : " << estLogDetQst << std::endl;
    
    nt_approx = 50; //floor(nt/10.0); //nt-2;
    Qst_approx.resize(nt_approx*ns, nt_approx*ns);
    fun->construct_Q_spat_temp_approx(theta, nt_approx, Qst_approx, estLogDetQst);
    std::cout << "nt : " << nt_approx << ", estLogDetQst   : " << estLogDetQst << std::endl;
    */

    /*
    nt_approx = 200; //floor(nt/10.0); //nt-2;
    Qst_approx.resize(nt_approx*ns, nt_approx*ns);
    fun->construct_Q_spat_temp_approx(theta, nt_approx, Qst_approx, estLogDetQst);
    std::cout << "nt : " << nt_approx << ", estLogDetQst   : " << estLogDetQst << std::endl;
    */

    double val;
    fun->eval_log_prior_lat(theta, val);
    std::cout << "nt : " << nt << ", true LogDet    : " << 2*val << std::endl;

    //std::cout << "\nnorm(Qst - Qst_approx) : " << (Qst - Qst_approx).norm() << std::endl;

    }

#endif // #if true/false


    double fx;

#if 0
	
    //theta_param << -2.152, 9.534, 11.927, 3.245;
   
    theta[0] = theta_param[0];
    fun->convert_interpret2theta_spatTemp(theta_param[1], theta_param[2], theta_param[3], theta[1], theta[2], theta[3]);

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

#if 1
    if(MPI_rank == 0)
        printf("\n====================== CALL BFGS SOLVER =====================\n");

    //LIKWID_MARKER_INIT;
    //LIKWID_MARKER_THREADINIT;

    //theta_param << -1.5, 7, 7, 3;
    //theta_param << -2.484481  7.836006  7.023295  2.504872
    //theta_param << -1.5, 8, 8, 3;

    //theta_param << -2.15, 9.57, 11.83, 3.24;

    theta[0] = theta_param[0];
    fun->convert_interpret2theta_spatTemp(theta_param[1], theta_param[2], theta_param[3], theta[1], theta[2], theta[3]);
    if(dim_th == 6){
        fun->convert_interpret2theta_spat(theta_param[4], theta_param[5], theta[4], theta[5]);
    }


    if(MPI_rank == 0){    
        std::cout << "theta param : " << theta_param.transpose() << std::endl;
        std::cout << "theta       : " << theta.transpose() << std::endl;
    }

    if(dim_th == 6){
        fun->convert_theta2interpret_spat(theta[4], theta[5], theta_param[4], theta_param[5]);
    }

    if(MPI_rank == 0){    
        std::cout << "theta param : " << theta_param.transpose() << std::endl;
        //std::cout << "theta       : " << theta.transpose() << std::endl;
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

    //exit(1);


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
        fun->convert_theta2interpret_spatTemp(theta_original[1], theta_original[2], theta_original[3], theta_original_param[1], theta_original_param[2], theta_original_param[3]);
        //std::cout << "\norig. mean interpret. param. : " << theta_original[0] << " " << prior_ranT << " " << prior_ranS << " " << prior_sigU << std::endl;
        std::cout << "\norig. mean interpret. param. : " << theta_original_param[0] << " " << theta_original_param[1] << " " << theta_original_param[2] << " " << theta_original_param[3] << std::endl;

        double lgamE = theta[1]; double lgamS = theta[2]; double lgamT = theta[3];
        double sigU; double ranS; double ranT;
        fun->convert_theta2interpret_spatTemp(lgamE, lgamS, lgamT, ranS, ranT, sigU);
        std::cout << "est.  mean interpret. param. : " << theta[0] << " " << ranS << " " << ranT << " " << sigU << std::endl;
    }

        // convert between different theta parametrisations
    if(dim_th == 6 && MPI_rank == 0){
        theta_original_param[0] = theta_original[0];
        fun->convert_theta2interpret_spatTemp(theta_original[1], theta_original[2], theta_original[3], theta_original_param[1], theta_original_param[2], theta_original_param[3]);
        fun->convert_theta2interpret_spat(theta_original[4], theta_original[5], theta_original_param[4], theta_original_param[5]);
        //std::cout << "\norig. mean interpret. param. : " << theta_original[0] << " " << prior_ranT << " " << prior_ranS << " " << prior_sigU << std::endl;
        std::cout << "\norig. mean interpret. param. : " << theta_original_param.transpose() << std::endl;

        double lgamE = theta[1]; double lgamS = theta[2]; double lgamT = theta[3];
        double sigU; double ranS; double ranT;
        fun->convert_theta2interpret_spatTemp(lgamE, lgamS, lgamT, ranS, ranT, sigU);
        fun->convert_theta2interpret_spat(theta[4], theta[5], theta_param[4], theta_param[5]);
        std::cout << "est.  mean interpret. param. : " << theta[0] << " " << ranS << " " << ranT << " " << sigU << " " << theta_param[4] << " " << theta_param[5] << std::endl;
    }

#endif

 double t_get_covariance = 0.0;

#if 1
    Vect theta_max(dim_th);
    //theta_max << -2.15, 9.57, 11.83, 3.24;    // theta
    //theta_max << 1.377415, -4.522942, 0.6501593, 1.710503, -4.603187, 2.243890;
    //theta_max << 1.374504, -4.442819,  0.672056,  1.592387, -4.366334,  2.014707;
    //theta_max << 1.385280, -4.425210, 0.6328134, 1.592043, -4.604329, 2.234010;
    theta_max = theta;

    // in what parametrisation are INLA's results ... ?? 
    double eps = 0.005;
    MatrixXd cov(dim_th,dim_th);

#if 0
    t_get_covariance = -omp_get_wtime();

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
    Vect interpret_theta(dim_th);
    interpret_theta[0] = theta_max[0];
    fun->convert_theta2interpret_spatTemp(theta_max[1], theta_max[2], theta_max[3], interpret_theta[1], interpret_theta[2], interpret_theta[3]);
    if(nss > 0){
        fun->convert_theta2interpret_spat(theta_max[4], theta_max[5], interpret_theta[4], interpret_theta[5]);
    }
    //interpret_theta << -2.152, 9.679, 12.015, 3.382;
//#ifdef PRINT_MSG 
    if(MPI_rank == 0){
        std::cout << "est. Hessian at theta param : " << interpret_theta.transpose() << std::endl;
    }
//#endif

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
    //theta_param << -1.407039,  8.841431,  9.956879,  3.770581;
    theta_param << -1.40701328482976, 9.34039748237832, 11.0020161941741, 4.27820007271347;
    theta[0] = theta_param[0];
    fun->convert_interpret2theta_spatTemp(theta_param[1], theta_param[2], theta_param[3], theta[1], theta[2], theta[3]);
    //theta << -1.407039, -7.801710, -6.339689, 5.588888;

    if(MPI_rank == 0){
        std::cout << "Computing mean latent parameters using theta interpret : " << theta_param.transpose() << std::endl;
    }
    */

    double t_get_fixed_eff;
    Vect mu(n);

    ArrayXi fact_to_rank_list(2);
    fact_to_rank_list << 0,0;
    /*if(MPI_size >= 2){
        fact_to_rank_list[1] = 1; 
    }*/

    if(MPI_rank == fact_to_rank_list[0] || MPI_rank == fact_to_rank_list[1]){
        //std::cout << "MPI rank = " << MPI_rank << ", fact_to_rank_list = " << fact_to_rank_list.transpose() << std::endl;
        std::cout << "Computing mu using theta : " << theta_max.transpose() << std::endl;
        t_get_fixed_eff = - omp_get_wtime();
        //fun->get_mu(theta, mu, fact_to_rank_list);
        fun->get_mu(theta_max, mu);
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

//#ifdef WRITE_RESULTS
    //std::string file_name_fixed_eff = results_folder + "/mean_latent_parameters.txt";
    std::string file_name_fixed_eff = "mean_latent_parameters.txt";
    write_vector(file_name_fixed_eff, mu, n);
//#endif

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

#if 1 

#ifdef PREDICT
    if(MPI_rank == 0){
    // =================================== compute prediction =================================== //

    // read in mu to check if its correct then ...
    //std::string mu_INLA_file = base_path + "/mu_INLA_" + to_string(n) + "_1.dat";
    //Vect mu_INLA = read_matrix(mu_INLA_file, n, 1);  

    // make prediction for mean at all locations y using Ax * mu
    std::cout << "dim(Ax_all) = " << Ax_all.rows() << " " << Ax_all.cols() << std::endl;
    std::cout << "mu(1:15) = " << mu.head(15).transpose() << std::endl;
    std::cout << "mu.tail(1:15) = " << mu.tail(15).transpose() << std::endl;
    Vect y_predict = Ax_all * mu; // mu_INLA
    Vect y_diff    = y_predict - y_all;

    int count_zeros = 0;
    for(int i=0; i<y_diff.size(); i++){
        // check if NA's in y_diff
        if(isnan(y_diff[i])){
            y_diff[i] = 0.0;
            count_zeros++;
        }
    }
    int nnz_y_diff = y_diff.size() - count_zeros;

    std::cout << "norm(y_diff) = " << 1.0/nnz_y_diff * y_diff.squaredNorm() << std::endl;
    std::cout << "y_predict(1:15) = " << y_predict.head(15).transpose() << std::endl;
    std::cout << "y(1:15) = " << y_all.head(15).transpose() << std::endl;

    std::cout << "y_predict.tail(1:15) = " << y_predict.tail(15).transpose() << std::endl;
    std::cout << "y.tail(1:15) = " << y_all.tail(15).transpose() << std::endl;

    // compute norm(Ax*mu - y) for y_ind = 0 & y_ind = 1 seperately -> check difference -> compare with INLA
    // consider first ntFit time steps
    Vect diff_y1(y_predict.size());
    diff_y1.array() =  y_diff.array() * y_ind.array();
    diff_y1 = diff_y1(seq(nt_init_fit, nt_last_fit));
    Vect y_ind_fit = y_ind(seq(nt_init_fit, nt_last_fit));
    std::cout << "norm(diff_y1) = " << 1.0 / y_ind_fit.sum() * diff_y1.squaredNorm() << std::endl;

    // have to remove NA's -> only consider known indices correponding to observed y's for nt_predict
    Vect diff_y0(y_predict.size());
    diff_y0.array() = y_diff.array() * (Vect::Ones(y_predict.size()) - y_ind).array();
    std::cout << "y_diff.size() - y_ind.sum()  = " << y_diff.size() - y_ind.sum() << std::endl;
    std::cout << "norm(diff_y0) = " << 1.0 / (y_diff.size() - y_ind.sum() ) * diff_y0.squaredNorm() << std::endl;

//#ifdef WRITE_RESULTS
        std::string file_name_y_predict_mean = "y_predict_mean_" + to_string(y_predict.size()) + ".txt";
        //std::string file_name_y_predict_mean = results_folder + "/y_predict_mean_" + to_string(y_predict.size()) + ".txt";
        // contains prediction for all y, not just previously unknown 
        // if only those needed, filter by indicator vector
        //printf("no all : %ld, rows(Ax_all) : %ld\n", no_all, Ax_all.rows());
        write_vector(file_name_y_predict_mean, y_predict, y_predict.size());
//#endif

    }

#endif // end predict

#endif // end #if 0/1

    } // end if(MPI_rank == fact_to_rank_list[1]), get_mu()

    // ============================================ validate ============================================= //
    if(validate && MPI_rank == 0){
        // compute 1/n*||(y - Ax*mu))|| for all w_i = 0 and w_i = 1 respectively
        std::cout << "sum(y) = " << y.sum() << std::endl;

        Vect diff_pred = y - Ax*mu;
        std::cout << "diff_pred(1:10) = " << diff_pred.head(10).transpose() << std::endl;

        double diff_temp_w1 = 0;
        double diff_temp_w0 = 0;

        for(int i=0; i<no; i++){
            if(w(i) == 1){
                diff_temp_w1 += diff_pred[i]*diff_pred[i];
            } else {
                diff_temp_w0 += diff_pred[i]*diff_pred[i];
            }
        }

        double diff_w1 = 1/w.sum()*sqrt(diff_temp_w1);
        double diff_w0 = 1/(no - w.sum())*sqrt(diff_temp_w0);

        std::cout << "difference w_i = 1 : " << diff_w1 << std::endl;
        std::cout << "difference w_i = 0 : " << diff_w0 << std::endl;
    }

  
    // =================================== compute marginal variances =================================== //
    double t_get_marginals = 0.0;

#if 1
    Vect marg(n);

    // when the range of u is large the variance of b0 is large.
    if(MPI_rank == 0){
        std::cout << "\n==================== compute marginal variances ================" << std::endl;
        //theta << -1.269613,  5.424197, -8.734293, -6.026165; // most common solution for temperature dataset
        std::cout << "\nUSING THETA : " << theta_max.transpose() << std::endl;


    	t_get_marginals = -omp_get_wtime();
    	fun->get_marginals_f(theta_max, marg);
    	t_get_marginals += omp_get_wtime();

        //std::cout << "\nest. variances fixed eff.    :  " << marg.tail(10).transpose() << std::endl;
        std::cout << "est. standard dev fixed eff  : " << marg.tail(nb).cwiseSqrt().transpose() << std::endl;
        std::cout << "est. std dev random eff      : " << marg.head(10).cwiseSqrt().transpose() << std::endl;
        //std::cout << "diag(Cov) :                     " << Cov.diagonal().transpose() << std::endl;

//#ifdef WRITE_RESULTS
    	std::string file_name_marg = "sd_latent_parameters.txt";
    	write_vector(file_name_marg, marg.cwiseSqrt(), n);
//#endif

#ifdef PREDICT

       // get marginal variances for all locations y using A*inv(Q)*A^T
       // TODOL QinvSp as rowmajor ...
       SpMat QinvSp(n,n);
       fun->get_fullFact_marginals_f(theta, QinvSp);

        std::cout << "norm(diag(invQ_fullMarg) - diag(marg)) = " << (QinvSp.diagonal() - marg).norm() << std::endl;

        /*
        std::string full_file_name = "Qinv_diag_" + to_string(n) + "_" + solver_type + ".txt";
        ofstream sol_file(full_file_name);
        if(sol_file){
            for (int i = 0; i < n; i++){
                sol_file << std::setprecision(7) << QinvSp.diagonal()[i] << "\n";
            }
            sol_file.close();
            std::cout << "wrote to file : " << full_file_name << std::endl;
        } else {
            std::cout << "There was an error writing " << full_file_name << " to file." << std::endl;
            exit(1);
        }
        std::cout << "QinvSp: est. standard dev fixed eff  : " << QinvSp.diagonal().tail(nb).cwiseSqrt().transpose() << std::endl;
        std::cout << "QinvSp: est. std dev random eff      : " << QinvSp.diagonal().head(10).cwiseSqrt().transpose() << std::endl;
        */

#if 1
    Vect projMargVar(Ax_all.rows());
    fun->compute_marginals_y(QinvSp, Ax_all, projMargVar);

    printf("Var theta0 = %f\n", 1/exp(theta_max[0]));
    printf("got here. nrows(Ax_all) = %d, no = %d,\n", Ax_all.rows(), no);
    // compute DS, CPRS, DCPRS scores in R ... just save relevant information to file
    Vect v = projMargVar + 1/exp(theta_max[0])*Vect::Ones(Ax_all.rows());
    std::cout << "projMargVar(1:10) = " << projMargVar.head(10).transpose() << std::endl;
    std::cout << "v(1:10) = " << v.head(10).transpose() << std::endl;
    // save results to file
    std::string file_name_v = "y_predict_v_" + to_string(v.size()) + ".txt"; //ntFit" + to_string(nt_fit) + "_ntPred" + to_string(nt_pred) + "_tInit" + to_string(nt_init_fit) + ".txt";
    write_vector(file_name_v, v, v.size());  

//#ifdef WRITE_RESULTS
    std::string file_name_sd = "y_predict_sd_" + to_string(projMargVar.size()) + ".txt"; // _ntFit" + to_string(nt_fit) + "_ntPred" + to_string(nt_pred) + "_tInit" + to_string(nt_init_fit) + ".txt";
    write_vector(file_name_sd, projMargVar.cwiseSqrt(), projMargVar.size());  
//#endif

#endif

#endif // end predict 

#if 0
       MatrixXd Qinv_full(n,n);
       fun->compute_fullInverseQ(theta, Qinv_full);
       Vect temp = QinvSp.diagonal() - marg;
       Vect temp2 = Qinv_full.diagonal() - marg;
       Vect temp3 = Qinv_full.diagonal() - QinvSp.diagonal();
       //std::cout << "norm(diag(QinvBlks) - marg) = " << temp.norm() << std::endl;
       std::cout << "norm(marg) = " << marg.norm() << ", norm(diag(Qinv_full)) = " << Qinv_full.diagonal().norm() << std::endl;
       std::cout << "norm(diag(Qinv_full) - marg) = " << temp2.norm() << std::endl;
       std::cout << "norm(diag(Qinv_full) - diag(QinvSp)) = " << temp3.norm() << std::endl;
       //std::cout << "norm(diag(Qinv_full) - marg) = " << temp2.head(50).transpose() << std::endl;

       MatrixXd Id(n,n);
       Id.setIdentity();
       SpMat Q(n,n);
       fun->construct_Q(theta, Q);
       std::cout << "norm(Qinv_full*Q - Id) = " << (Qinv_full*Q - Id).norm() << std::endl;

        std::string file_Qinv_full_diag = "Qinv_full_diag_" + to_string(n) + "_" + solver_type + ".txt";
        write_vector_lowerPrec(file_Qinv_full_diag, Qinv_full.diagonal(), n);

        std::string file_Qinv_diag = "Qinv_diag_" + to_string(n) + "_" + solver_type + ".txt";
        write_vector_lowerPrec(file_Qinv_diag, marg, n);

        std::string file_Qinv_selInv_diag = "Qinv_selInv_diag_" + to_string(n) + "_" + solver_type +  ".txt";
        write_vector_lowerPrec(file_Qinv_selInv_diag, QinvSp.diagonal(), n);

       //std::cout << "Qinv_full - QinvSp: " << Qinv_full.block(0,0,20,20) - QinvSp.block(0,0,20,20) << std::endl;

       //MatrixXd QinvProjFULL = Ax_all*Qinv_full*Ax_allT;
       //Vect projMargVarFULL = QinvProjFULL.diagonal();
       //std::cout << "est. std dev 1st 10 loc FULL       : " << projMargVarFULL.head(20).cwiseSqrt().transpose() << std::endl;
#endif


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
