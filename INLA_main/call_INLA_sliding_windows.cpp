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
//#define PRINT_MSG
//#define WRITE_LOG

#include "mpi.h"

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

using namespace LBFGSpp;

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
    int whatever = 2;

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
    
    if(argc != 1 + 8 && MPI_rank == 0){
        std::cout << "wrong number of input parameters. " << std::endl;

        std::cerr << "INLA Call : ns nt nss nb noPerTs path/to/files solver_type" << std::endl;

        std::cerr << "[integer:ns]                number of spatial grid points " << std::endl;
        std::cerr << "[integer:nt]                number of temporal grid points " << std::endl;
        std::cerr << "[integer:nss]               number of spatial grid points add. spatial field" << std::endl;
        std::cerr << "[integer:nb]                number of fixed effects" << std::endl;
        std::cerr << "[integer:noPerTs]           number of data samples per time step" << std::endl;

        std::cerr << "[string:likelihood]         Gaussian/Poisson/Binomial" << std::endl;
        std::cerr << "[string:base_path]          path to folder containing matrix files " << std::endl;
        std::cerr << "[string:solver_type]        BTA or PARDISO or Eigen" << std::endl;

        exit(1);
    }

#ifdef PRINT_MSG
    if(MPI_rank == 0){
        std::cout << "reading in example. " << std::endl;
    }
#endif

    size_t ns  = atoi(argv[1]);
    size_t nt  = atoi(argv[2]);
    size_t nss = atoi(argv[3]);
    size_t nb  = atoi(argv[4]);
    size_t noPerTs  = atoi(argv[5]);
    size_t no = noPerTs * nt;

    // set nt = 1 if ns > 0 & nt = 0
    if(ns > 0 && nt == 0){
        nt = 1;
    }else if(ns == 0){
        nt = 0;
    }

    size_t n = ns*nt + nss + nb;

    // also save as string
    std::string ns_s = std::to_string(ns);
    std::string nt_s = std::to_string(nt);
    std::string nb_s = std::to_string(nb);
    std::string no_s = std::to_string(no); 
    std::string n_s  = std::to_string(n);

    std::string likelihood  = argv[6];
    std::string base_path   = argv[7];    
    std::string solver_type = argv[8];

    // check likelihood among the available options
    if(likelihood.compare("Gaussian") == 0        || likelihood.compare("gaussian") == 0){
        likelihood = "gaussian";
    } else if(likelihood.compare("Poisson") == 0  || likelihood.compare("poisson")  == 0){
        likelihood = "poisson";
    } else if(likelihood.compare("Binomial") == 0 || likelihood.compare("binomial") == 0 ){
        likelihood = "binomial";
    } else {
        std::cout << "unknown likelihood: " << likelihood << std::endl;
        exit(1);
    }

    if(MPI_rank == 0){
        std::cout << "Assumed likelihood of the data: " << likelihood << std::endl;
    }

    // check if solver type is neither PARDISO nor BTA :
    if(solver_type.compare("PARDISO") != 0 && solver_type.compare("BTA") != 0 && solver_type.compare("Eigen") != 0){
        std::cout << "Unknown solver type. Available options are :\nPARDISO\nBTA\nEigen" << std::endl;
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

    // ======================================================================================================== //
    // read in sliding window parameters
    int no_moving_windows = 1;
    //std::string moving_window_dim_file        =  base_path + "/temporal_moving_windows_info_" + std::to_string(no_moving_windows) + "_5.dat";
    //file_exists(moving_window_dim_file); 
    //MatrixXd moving_window_dim = read_matrix(moving_window_dim_file, no_moving_windows, 5);

    // decide this manually
    MatrixXd moving_window_dim = MatrixXd::Zero(no_moving_windows, 5);
    int nt_subset = 30; // length of each moving window
    int offset = 5; // offset between moving windows
    moving_window_dim(0,0) = 0; moving_window_dim(0,1) = 1; moving_window_dim(0,2) = nt_subset; moving_window_dim(0,3) = 1; moving_window_dim(0,4) = noPerTs * nt_subset;
    //moving_window_dim(1,0) = 1; moving_window_dim(1,1) = offset; moving_window_dim(1,2) = offset + nt_subset; moving_window_dim(1,3) = moving_window_dim(1,1) * noPerTs + 1; moving_window_dim(1,4) = moving_window_dim(1,2) * noPerTs;
    //moving_window_dim(2,0) = 2; moving_window_dim(2,1) = 2*offset; moving_window_dim(2,2) = 2*offset + nt_subset; moving_window_dim(2,3) = moving_window_dim(2,1) * noPerTs + 1; moving_window_dim(2,4) = moving_window_dim(2,2) * noPerTs;

    // if(moving_window_dim(2,2) > nt){
    //     std::cout << "Incompatible moving window dimensions detected." << std::endl;
    //     printf("nt = %d, but last moving window ends at time step %f\n", nt, moving_window_dim(2,2));
    //     exit(1);
    // }

    if(MPI_rank == 0){
        std::cout << "Moving window dimensions : \n" << moving_window_dim << std::endl;
    }

    /* ---------------- read in matrices ---------------- */
    // dimension hyperparamter vector
    int dim_th;
    int dim_spatial_domain = 2;
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
    SpMat Ax; 
    Vect y;

    // not needed for Gaussian case. Initialize to zero otherwise ...
    Vect mu_initial = Vect::Zero(n);

    // additional coefficient vector: Poisson / Binomial data
    Vect extraCoeffVecLik;

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

    Vector3i dimList = Vector3i::Zero(3);

    if(MPI_rank == 0){
        printf("\n==================== MODEL SPECIFICATIONS ===================\n");
    }

    if(likelihood.compare("gaussian") == 0){
        dimList(0) = 1;
    }

    if(ns == 0){

        if(likelihood.compare("gaussian") == 0){
            dim_th = 1;
        } else {
            dim_th = 0;
        }

        // read in design matrix 
        // files containing B
        std::string B_file        =  base_path + "/A_" + no_s + "_" + nb_s + ".dat";
        file_exists(B_file); 

        if(MPI_rank == 0){
            std::cout << "total number of observations : " << no << std::endl;
        }

        B = read_matrix(B_file, no, nb);
        Ax = B.sparseView(); // so that I can use Ax for regression case as well

        //std::cout << "y : \n"  << y << std::endl;    
        //std::cout << "B : \n" << B << std::endl;

    } else if(ns > 0 && nt == 1){

        if(MPI_rank == 0){
            std::cout << "spatial model." << std::endl;
        }

        dimList(2) = 2;

        if(likelihood.compare("gaussian") == 0){
            dim_th = 3;
        } else {
            dim_th = 2;          
        }

        // check spatial FEM matrices
        std::string c0_file       =  base_path + "/c0_" + ns_s + ".dat";
        file_exists(c0_file);
        std::string g1_file       =  base_path + "/g1_" + ns_s + ".dat";
        file_exists(g1_file);
        std::string g2_file       =  base_path + "/g2_" + ns_s + ".dat";
        file_exists(g2_file);

        // check projection matrix for A.st
        // std::string Ax_file     =  base_path + "/Ax_" + no_s + "_" + n_s + ".dat";
        // file_exists(Ax_file);

        // read in matrices
        c0 = read_sym_CSC(c0_file);
        g1 = read_sym_CSC(g1_file);
        g2 = read_sym_CSC(g2_file);

        // doesnt require no to be read, can read no from Ax
        //Ax = readCSC(Ax_file);
        // get rows from the matrix directly
        // doesnt work for B
        //no = Ax.rows();

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

        dimList(1) = 3;
        if(likelihood.compare("gaussian") == 0){
            dimList(0) = 1;
        }

        if(nss > 0){
            dimList(2) = 2;
        }
        dim_th = dimList.sum();

        //std::cout << "likelihood : " << likelihood << std::endl;
        
        if(MPI_rank == 0){
            printf("spatial-temporal model");
            if(nss > 0){
                printf(" with add. spatial field");
            }
            printf(". dim(theta) = %d\n", dim_th);
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

        // check projection matrix for A.st
        // std::string Ax_file     =  base_path + "/Ax_" + no_s + "_" + n_s + ".dat";
        // file_exists(Ax_file);

        // read in matrices
        c0 = read_sym_CSC(c0_file);
        g1 = read_sym_CSC(g1_file);
        g2 = read_sym_CSC(g2_file);
        g3 = read_sym_CSC(g3_file);

        // M matrices and total Ax removed in favor of loop

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

    // data y
    std::string y_file        =  base_path + "/y_" + no_s + "_1" + ".dat";
    file_exists(y_file);

    y = read_matrix(y_file, no, 1);  
    if(MPI_rank == 0){ 
        std::cout << "sum(y) = " << y.sum() << std::endl;
    }

    Vect mean_latent_original(n);
    if(likelihood.compare("poisson") == 0 || likelihood.compare("binomial") == 0){
        // TODO: something like if does not exist assume all ONES ??
        int no_subset = noPerTs * nt_subset;
        std::string extraCoeffVecLik_file        =  base_path + "/extraCoeff_" + to_string(no_subset) + "_1" + ".dat";
        file_exists(extraCoeffVecLik_file);
        extraCoeffVecLik = read_matrix(extraCoeffVecLik_file, no_subset, 1);

        //std::string mean_latent_file        =  base_path + "/mean_latent_original_" + to_string(n) + "_1" + ".dat";
        //std::string mean_latent_file        =  base_path + "/mean_latent_INLA_" + to_string(n) + "_1" + ".dat";

        //file_exists(mean_latent_file);
        //mean_latent_original = read_matrix(mean_latent_file, n, 1);  
        mean_latent_original.setZero(n);

        // TODO: some initial guesses seem to be working others not ... whats going on
        mu_initial = mean_latent_original; // + 0.5 * Vect::Random(n);

        if(MPI_rank == 0){
            std::cout << "extraCoeffVecLik: " << extraCoeffVecLik.head(10).transpose() << std::endl;    
            std::cout << "mean latent original: " << mean_latent_original.head(min(10, int (n))).transpose() << std::endl;    
            std::cout << "mu initial : " << mu_initial.head(min(10, int (n))).transpose() << std::endl;    
        }
    }

#ifdef PRINT_MSG
    std::cout << "dim(c0) = " << c0.rows() << " x " << c0.cols() << std::endl;
    std::cout << "dim(g1) = " << g1.rows() << " x " << g1.cols() << std::endl;
    std::cout << "dim(g2) = " << g2.rows() << " x " << g2.cols() << std::endl;
    std::cout << "dim(g3) = " << g3.rows() << " x " << g3.cols() << std::endl;
    std::cout << "dim(M0) = " << M0.rows() << " x " << M0.cols() << std::endl;
    std::cout << "dim(M1) = " << M1.rows() << " x " << M1.cols() << std::endl;
    std::cout << "dim(M2) = " << M2.rows() << " x " << M2.cols() << std::endl;
    std::cout << "dim(Ax) = " << Ax.rows() << " x " << Ax.cols() << std::endl;
    std::cout << "dim(y) = " << y.rows() << " x " << y.cols() << std::endl;
#endif

    /* ----------------------- initialise random theta -------------------------------- */

    //Vect theta(dimList.sum()); 
    Vect theta(dim_th); theta.setZero();  // define initial guess in model parametrization
    Vect theta_param(dim_th); theta_param.setZero();     // or in interpretable parametrization
    Vect theta_prior_param(dim_th);
    Vect theta_original(dim_th); theta_original.setZero();
    Vect theta_original_param(dim_th); theta_original_param.setZero();

    std::string data_type;

    // initialise theta
    if(ns == 0 && nt == 0 && likelihood.compare("gaussian") == 0){

        // Initial guess
        theta[0] = 3;
        theta_param[0] = theta[0];
        theta_prior_param[0] = theta_original[0];

        std::string theta_original_param_file        =  base_path + "/theta_original_" + to_string(dim_th) + "_1" + ".dat";
        file_exists(theta_original_param_file); 
        theta_original = read_matrix(theta_original_param_file, dim_th, 1);

        if(MPI_rank == 0){
            std::cout << "initial theta : "  << theta.transpose() << std::endl; 
        }   
    } else if(ns == 0 && nt == 0 && likelihood.compare("poisson") == 0){
        printf("Poisson regression. No hyperparameters needed!\n");

    } else if(ns > 0 && nt == 1 ){ // && likelihood.compare("gaussian") != 0
        if(MPI_rank == 0){ 
            std::cout << "using SYNTHETIC DATASET" << std::endl; 
        }
        dim_spatial_domain = 2; 
        // read in original theta for comparison. order: prec obs, range s, prec sigma u
        std::string theta_original_param_file        =  base_path + "/theta_interpretS_original_" + to_string(dim_th) + "_1" + ".dat";
        //std::string theta_original_param_file        =  base_path + "/theta_interpretS_INLA_" + to_string(dim_th) + "_1" + ".dat";
        file_exists(theta_original_param_file); 
        theta_original_param = read_matrix(theta_original_param_file, dim_th, 1);

        // read in lambdas for pc prior. same order.  
        std::string lambda_file        =  base_path + "/pc_prior_theta_lambdas_" + to_string(dim_th) + "_1" + ".dat";
        file_exists(lambda_file);

        theta_prior_param = read_matrix(lambda_file, dim_th, 1);

        theta_param << theta_original_param + 2*Vect::Random(dim_th);
        if(MPI_rank == 0){  
            std::cout << "initial theta param : "  << theta_param.transpose() << std::endl;  
        } 

    } else if (ns > 0 && nt > 1){

        if(likelihood.compare("gaussian") == 0){

#ifdef DATA_SYNTHETIC
            data_type = "synthetic";

            // constant in conversion between parametrisations changes dep. on spatial dim
            // assuming sphere -> assuming R^3
            dim_spatial_domain = 2;
            //theta_test.spatial_dim = theta_prior_test.spatial_dim = theta_original_test.spatial_dim = 2;

            // define if on the sphere or plane
            manifold = "sphere";
            //theta_test.manifold = theta_prior_test.manifold = theta_original_test.manifold = "sphere"; 

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
                theta_original_param << 1.38629400, -0.00000023, 2.30258418, 1.40625832;

                // using PC prior, choose lambda  
                theta_prior_param << 0.7/3.0, 0.2*0.7*0.7, 0.7, 0.7/3.0;

                //theta_param << 1.373900, 2.401475, 0.046548, 1.423546; 
                //theta_param << 4, 0, 0, 0;
                theta_param << 1.366087, 2.350673, 0.030923, 1.405511;
                //theta_test.update_interpretS(theta_param);

            } else {
                // order prec obs, lgamS for st , lgamT for st, lgamE for st, lgamE for s, lgamS for s
                theta_original       << 1.386294, -3.870213, 0.6342557, 1.961659, -1.206621, -0.05889152;
                theta_original_param << 1.386, 0.405, 1.099, 1.386, -1.204, 1.099;
                //theta_prior_param     << 1.386294,     -4.469624,      0.6342557,    1.673976, -4.607818, 2.243694;
                // order: prec obs, range s for st, range t for st, prec sigma for st, range s for s, prec sigma for s
                //theta_prior_param  << -log(0.01)/5, -log(0.01)*0.1, -log(0.01)*1, -log(0.01)/1, -log(0.01)*(3000.0/6371.0), -log(0.01)/5;
                theta_prior_param  << -log(0.01)/5, -log(0.01)*pow(0.1, 0.5*dim_spatial_domain), -log(0.01)*pow(1, 0.5), -log(0.01)/1,-log(0.01)*pow(3000.0/6371.0, 0.5*dim_spatial_domain), -log(0.01)/5;
                
                if(MPI_rank == 0){
                    std::cout << "theta prior param : " << theta_prior_param.transpose() << std::endl;
                }
                // same order as above
                //theta_param << 1.386294, 0.4054651, 1.386294,  0.6931472, 1.098612,  0.000000;
                //theta_param        << 1.033, 0.431, 0.756, 0.247, 1.608, -0.495;
                theta_param <<  4, 1, 3, 2, -1, 0;
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
                    int i_end   = min(tsPerConstr*(i+1), (int) nt);
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
           
            } // end if(constraint)

#elif defined(DATA_TEMPERATURE)

            // =========== temperature data set =============== //
            data_type = "temperature";

            if(MPI_rank == 0){
                std::cout << "using TEMPERATURE DATASET" << std::endl; 
                if(constr)
                    std::cout << "assuming sum-to-zero constraints on spatial-temporal field." << std::endl;
            }

            // constant in conversion between parametrisations changes dep. on spatial dim
            dim_spatial_domain = 2;
            //theta_test.spatial_dim = theta_prior_test.spatial_dim = theta_original_test.spatial_dim = 2;

            //theta_param << 4, 0, 0, 0;
            //theta_param << -1.5, 7, 7, 3;
            theta_param << 0, 7, 2, 1.4;
            //theta_test.update_interpretS(theta_param);

            //theta_original << -1.269613,  5.424197, -8.734293, -6.026165;  // estimated from INLA / same for my code varies a bit according to problem size
            //theta_original_param << -2.090, 9.245, 11.976, 2.997; // estimated from INLA

            // using PC prior, choose lambda  
            // previous order : interpret_theta & lambda order : sigma.e, range t, range s, sigma.u -> doesn't match anymore
            // NEW ORDER sigma.e, range s, range t, sigma.u 
            //theta_prior_param << 0.7/3.0, 0.2*0.7*0.7, 0.7, 0.7/3.0;
            // -log(p)/u where c(u, p)
            theta_prior_param[0] = -log(0.01)/5; 	      //prior.sigma obs : 5, 0.01
            //theta_prior_param[1] = -log(0.5)/1000;        //prior.rs=c(1000, 0.5), ## P(range_s < 1000) = 0.5
            theta_prior_param[1] = -log(0.01)/300;        
            //theta_prior_param[2] = -log(0.5)/20;	      //prior.rt=c(20, 0.5), ## P(range_t < 20) = 0.5
            theta_prior_param[2] = -log(0.01)/1;
            //theta_prior_param[3] = -log(0.5)/10;          //prior.sigma=c(10, 0.5) ## P(sigma_u > 10) = 0.5
            theta_prior_param[3] = -log(0.01)/5;	    
            //theta_prior_test.update_interpretS(theta_prior_param);

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
                    int i_end   = std::min(tsPerConstr*(i+1), (int) nt);
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

        // none-gaussian likelihood
        } else {

            if(MPI_rank == 0){ 
                std::cout << "Spatial-Temporal model." << std::endl;
                std::cout << "using SYNTHETIC DATASET" << std::endl; 
            }
            dim_spatial_domain = 2; 
            // read in original theta for comparison. order: prec obs, range s, prec sigma u
            std::string theta_original_param_file        =  base_path + "/theta_init_" + to_string(dim_th) + "_1" + ".dat";
            //std::string theta_original_param_file        =  base_path + "/theta_interpretS_original_" + to_string(dim_th) + "_1" + ".dat";
            //std::string theta_original_param_file        =  base_path + "/theta_interpretS_INLA_" + to_string(dim_th) + "_1" + ".dat";
            file_exists(theta_original_param_file); 
            theta_original_param = read_matrix(theta_original_param_file, dim_th, 1);

            // read in lambdas for pc prior. same order.  
            std::string lambda_file        =  base_path + "/pc_prior_theta_lambdas_" + to_string(dim_th) + "_1" + ".dat";
            file_exists(lambda_file);

            theta_prior_param = read_matrix(lambda_file, dim_th, 1);

            //theta_prior_param << 1, -2.3, 2.1;
            theta_param << theta_original_param; // + 2*Vect::Random(dim_th);
            if(MPI_rank == 0){
                std::cout << "initial theta param : "  << theta_param.transpose() << std::endl; 
            }

        } // end if : gaussian / non-gaussian

    } else {
        printf("unknown parameters, likelihood or datatype!");
        std::cout << "likelihood : " << likelihood << std::endl;
        exit(1);
    } // end else for spatial-temporal case

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

 //#if 1
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
    param.delta = 1e-3;
    //param.delta = 1e-7;
    //param.delta = 1e-9; // ref sol
    // maximum line search iterations
    param.max_iterations = 200; //200;

    // Create solver and function object
    LBFGSSolver<double> solver(param);

    // ===========================================================================================================================================
    // ===========================================================================================================================================
    // ===========================================================================================================================================

    SpMat Ax_window;
    
    // LOOP FROM HERE ONWARDS ... SLIDING WINDOWS
    int num_windows = moving_window_dim.rows();
    for (int window = 0; window < num_windows; window++){

        // ============================ set up Posterior of theta ======================== //
        // compute indices
        int nt_start = moving_window_dim(window,1) - 1;
        int nt_end   = moving_window_dim(window,2) - 1;
        int nt_subset = nt_end - nt_start + 1;

        int n_subset_start = ns*nt_start;
        int n_subset_end   = ns*(nt_end+1) - 1;
        int n_subset = n_subset_end - n_subset_start + 1 + nb;

        if(MPI_rank == 0){
            std::cout << "\n=========== window " << window+1 << " / " << num_windows << ". time points: " << nt_start << " - " << nt_end << " ======\n" << std::endl;
            std::cout << "DEBUG: Window dimensions calculation:" << std::endl;
            std::cout << "  ns = " << ns << ", nt = " << nt << ", nb = " << nb << ", no = " << no << std::endl;
            std::cout << "  nt_start = " << nt_start << ", nt_end = " << nt_end << ", nt_subset = " << nt_subset << std::endl;
            std::cout << "  n_subset_start = " << n_subset_start << ", n_subset_end = " << n_subset_end << std::endl;
            std::cout << "  n_subset = " << n_subset << " (should be " << (n_subset_end - n_subset_start + 1 + nb) << ")" << std::endl;
            std::cout << "  Full n = " << n << " (should be " << (ns*nt + nb) << ")" << std::endl;
        }

        int obs_start = moving_window_dim(window,3) - 1;
        int obs_end   = moving_window_dim(window,4) - 1;
        int no_subset = obs_end - obs_start + 1;

        if(MPI_rank == 0){
            std::cout << "DEBUG: Observation window dimensions:" << std::endl;
            std::cout << "  obs_start = " << obs_start << ", obs_end = " << obs_end << std::endl;
            std::cout << "  no_subset = " << no_subset << " (should be " << (obs_end - obs_start + 1) << ")" << std::endl;
        }

        Vect y_window = y.segment(obs_start, obs_end - obs_start + 1);
        Vect extraCoeffVecLik_window = extraCoeffVecLik.segment(obs_start, obs_end - obs_start + 1);
        
        if(MPI_rank == 0){
            std::cout << "DEBUG: Vector segment extraction:" << std::endl;
            std::cout << "  y.size() = " << y.size() << ", y_window.size() = " << y_window.size() << std::endl;
            std::cout << "  extraCoeffVecLik.size() = " << extraCoeffVecLik.size() << ", extraCoeffVecLik_window.size() = " << extraCoeffVecLik_window.size() << std::endl;
            std::cout << "  mu_initial.size() = " << mu_initial.size() << std::endl;
            std::cout << "  mean_latent_original.size() = " << mean_latent_original.size() << std::endl;
        }
        
        // Create window-specific mu_initial (spatial+temporal subset + regression coefficients)
        if(MPI_rank == 0){
            std::cout << "DEBUG: Creating mu_initial_window:" << std::endl;
            std::cout << "  Attempting to extract segment from mu_initial[" << n_subset_start << ":" << (n_subset_start + n_subset_end - n_subset_start) << "]" << std::endl;
            std::cout << "  Segment size needed: " << (n_subset_end - n_subset_start + 1) << std::endl;
            if(n_subset_start + (n_subset_end - n_subset_start + 1) > mu_initial.size()) {
                std::cout << "  ERROR: Segment extraction would exceed mu_initial bounds!" << std::endl;
                std::cout << "  mu_initial.size() = " << mu_initial.size() << std::endl;
                std::cout << "  Trying to access up to index " << (n_subset_start + n_subset_end - n_subset_start) << std::endl;
                std::cout << "FATAL: Exiting due to out-of-bounds segment extraction!" << std::endl;
                exit(1);
            }
        }
        
        Vect mu_initial_window(n_subset);
        mu_initial_window.head(n_subset_end - n_subset_start + 1) = mu_initial.segment(n_subset_start, n_subset_end - n_subset_start + 1);
        mu_initial_window.tail(nb) = mu_initial.tail(nb);
        
        // Create window-specific mean_latent_original (spatial+temporal subset + regression coefficients)
        if(MPI_rank == 0){
            std::cout << "DEBUG: Creating mean_latent_original_window:" << std::endl;
            std::cout << "  mu_initial_window.size() = " << mu_initial_window.size() << std::endl;
        }
        
        Vect mean_latent_original_window(n_subset);
        mean_latent_original_window.head(n_subset_end - n_subset_start + 1) = mean_latent_original.segment(n_subset_start, n_subset_end - n_subset_start + 1);
        mean_latent_original_window.tail(nb) = mean_latent_original.tail(nb);
        
        if(MPI_rank == 0){
            std::cout << "  mean_latent_original_window.size() = " << mean_latent_original_window.size() << std::endl;
        }

        // read in temporal matrices for the current window 
        std::string M0_file      =  base_path + "/M0_" + std::to_string(nt_subset) + ".dat";
        file_exists(M0_file);
        std::string M1_file      =  base_path + "/M1_" + std::to_string(nt_subset) + ".dat";
        file_exists(M1_file);
        std::string M2_file      =  base_path + "/M2_" + std::to_string(nt_subset) + ".dat";
        file_exists(M2_file);  

        // read in files
        SpMat M0_window = read_sym_CSC(M0_file);
        SpMat M1_window = read_sym_CSC(M1_file);
        SpMat M2_window = read_sym_CSC(M2_file);

        // read in Ax for current window
        if(ns > 0){
            std::string Ax_file_window = base_path + "/Ax_" + to_string(no_subset) + "_" + to_string(n_subset) + ".dat";
            //std::string Ax_file_window = base_path + "/Ax_window" + to_string(window+1) + "_" +  to_string(no_subset)  + "_" +to_string(n_subset) + ".dat";
            if(MPI_rank == 0){
                std::cout << "DEBUG: Reading Ax_window from file: " << Ax_file_window << std::endl;
            }
            file_exists(Ax_file_window);
            Ax_window = readCSC(Ax_file_window);
            
            if(MPI_rank == 0){
                std::cout << "DEBUG: Ax_window loaded successfully:" << std::endl;
                std::cout << "  Ax_window.rows() = " << Ax_window.rows() << " (expected: " << no_subset << ")" << std::endl;
                std::cout << "  Ax_window.cols() = " << Ax_window.cols() << " (expected: " << n_subset << ")" << std::endl;
                std::cout << "  Ax_window.nonZeros() = " << Ax_window.nonZeros() << std::endl;
                
                if(Ax_window.rows() != no_subset || Ax_window.cols() != n_subset) {
                    std::cout << "ERROR: Ax_window dimensions don't match expected!" << std::endl;
                    std::cout << "  Expected: " << no_subset << " x " << n_subset << std::endl;
                    std::cout << "  Actual: " << Ax_window.rows() << " x " << Ax_window.cols() << std::endl;
                    std::cout << "FATAL: Exiting due to Ax_window dimension mismatch!" << std::endl;
                    exit(1);
                }
            }
        }

        //std::optional<PostTheta> fun;
        PostTheta* fun;

        if(ns == 0 && nss == 0){
            // fun.emplace(nb, no, B, y);
            if(MPI_rank == 0){
                std::cout << "DEBUG: About to call regression model constructor with:" << std::endl;
                std::cout << "  ns=" << ns << ", nt_subset=" << nt_subset << ", nb=" << nb << ", no_subset=" << no_subset << std::endl;
                std::cout << "  B.rows()=" << B.rows() << ", B.cols()=" << B.cols() << std::endl;
                std::cout << "  y_window.size()=" << y_window.size() << std::endl;
                std::cout << "  mu_initial_window.size()=" << mu_initial_window.size() << std::endl;
                std::cout << "Call constructor for regression model." << std::endl;
            }
            fun = new PostTheta(ns, nt_subset, nb, no_subset, B, y_window, theta_prior_param, mu_initial_window, likelihood, extraCoeffVecLik_window, solver_type, constr, Dxy, validate, w);
        } else if(ns > 0 && nt == 1 && nss == 0) {
            if(MPI_rank == 0){
                std::cout << "DEBUG: About to call spatial constructor with:" << std::endl;
                std::cout << "  ns=" << ns << ", nt_subset=" << nt_subset << ", nb=" << nb << ", no_subset=" << no_subset << std::endl;
                std::cout << "  Ax_window.rows()=" << Ax_window.rows() << ", Ax_window.cols()=" << Ax_window.cols() << std::endl;
                std::cout << "  y_window.size()=" << y_window.size() << std::endl;
                std::cout << "  mu_initial_window.size()=" << mu_initial_window.size() << std::endl;
                std::cout << "\ncall spatial constructor." << std::endl;
            }
            // PostTheta fun(nb, no, B, y);
            fun = new PostTheta(ns, nt_subset, nb, no_subset, Ax_window, y_window, c0, g1, g2, theta_prior_param, mu_initial_window, likelihood, extraCoeffVecLik_window, solver_type, dim_spatial_domain, manifold, constr, Dx, Dxy, validate, w);
        } else if(ns > 0 && nt > 1 && nss == 0){
            if(MPI_rank == 0){
                std::cout << "DEBUG: About to call spatial-temporal constructor with:" << std::endl;
                std::cout << "  ns=" << ns << ", nt_subset=" << nt_subset << ", nb=" << nb << ", no_subset=" << no_subset << std::endl;
                std::cout << "  Ax_window.rows()=" << Ax_window.rows() << ", Ax_window.cols()=" << Ax_window.cols() << std::endl;
                std::cout << "  y_window.size()=" << y_window.size() << std::endl;
                std::cout << "  mu_initial_window.size()=" << mu_initial_window.size() << std::endl;
                std::cout << "\ncall spatial-temporal constructor." << std::endl;
                std::cout << "likelihood: " << likelihood << std::endl;
            }
            fun = new PostTheta(ns, nt_subset, nb, no_subset, Ax_window, y_window, c0, g1, g2, g3, M0_window, M1_window, M2_window, theta_prior_param, mu_initial_window, likelihood, extraCoeffVecLik_window, solver_type, dim_spatial_domain, manifold, constr, Dx, Dxy, validate, w);
        } else if(ns > 0 && nt > 1 && nss > 0){
            if(MPI_rank == 0){
                std::cout << "DEBUG: About to call spatial-temporal + additional spatial constructor with:" << std::endl;
                std::cout << "  ns=" << ns << ", nt_subset=" << nt_subset << ", nss=" << nss << ", nb=" << nb << ", no_subset=" << no_subset << std::endl;
                std::cout << "  Ax_window.rows()=" << Ax_window.rows() << ", Ax_window.cols()=" << Ax_window.cols() << std::endl;
                std::cout << "  y_window.size()=" << y_window.size() << std::endl;
                std::cout << "  mu_initial_window.size()=" << mu_initial_window.size() << std::endl;
                std::cout << "\ncall spatial-temporal constructor with add. spatial field." << std::endl;
            } 
            fun = new PostTheta(ns, nt_subset, nss, nb, no_subset, Ax_window, y_window, c0, g1, g2, g3, M0, M1, M2, theta_prior_param, mu_initial_window, likelihood, extraCoeffVecLik_window, solver_type, dim_spatial_domain, manifold, constr, Dx, Dxy, validate, w);
        } else {
            printf("invalid combination of parameters!\n");
            printf("ns = %ld, nt = %ld, nss = %ld\n", ns, nt, nss);
            exit(1);
        }

        if(MPI_rank == 0){
            std::cout << "DEBUG: PostTheta constructor completed successfully!" << std::endl;
        }

        printf("after constructor\n");
    
        Vect theta_test(dim_th);
        Vect theta_param_test(dim_th);
        fun->convert_interpret2theta(theta_param, theta_test);
        fun->convert_theta2interpret(theta_test, theta_param_test);

        if(MPI_rank == 0){
            printf("\n======================= HYPERPARAMETERS =====================\n");
        }

        if(MPI_rank == 0){
            // TO BE deleted later
            std::cout << "theta prior param       : " << theta_prior_param.transpose() << std::endl;
            //std::cout << "theta orig. param  : " << theta_original_param.transpose() << std::endl;
            //std::cout << "theta original     : " << std::right << std::fixed << theta_original.transpose() << std::endl;
            //std::cout << "theta prior param  : " << theta_prior_test.flatten_modelS().transpose() << std::endl;
        }

        // convert from interpretable parametrisation to internal one
        fun->convert_interpret2theta(theta_param, theta);
        if(MPI_rank == 0){
            std::cout << "theta interpret. param. : "  << std::right << std::fixed << theta_param.transpose() << std::endl;
            std::cout << "initial theta                   : "  << std::right << std::fixed << theta.transpose() << std::endl;       
        }

        fun->convert_interpret2theta(theta_original_param, theta_original);
        /*if(MPI_rank == 0){
            std::cout << "theta original param.         : "  << std::right << std::fixed << theta_original_param.transpose() << std::endl;
            std::cout << "original theta                   : "  << std::right << std::fixed << theta_original.transpose() << std::endl;       
        }*/

        if(MPI_rank == 0 && likelihood.compare("gaussian") != 0){
            std::cout << "\n================== Testing inner Iteration. =====================\n" << std::endl;

            // read in original latent parameters
            /*std::string beta_file        =  base_path + "/beta_original_" + to_string(nb) + "_1" + ".dat";
            file_exists(beta_file);
            Vect beta_original = read_matrix(beta_file, nb, 1);  
            std::cout << "beta original: " << beta_original.transpose() << std::endl;*/

            // Use the already created extraCoeffVecLik_window instead of reading from file
            std::cout << "extraCoeffVecLik: " << extraCoeffVecLik_window.head(10).transpose() << std::endl;

            // no separate function to construct Qprior
            SpMat Qprior(n_subset,n_subset);
            fun->get_Qprior(theta_original, Qprior);
            std::cout << "Qprior(1:10,1:10) = \n" << Qprior.block(0, 0, min(10, (int) n), min(10, (int) n)) << std::endl;
            //std::cout << "Qprior(1:10,1:10) = \n" << Qprior.block(399, 399, 30, 30) << std::endl;

            double val_logPriorLat = fun->cond_LogPriorLat(Qprior, mean_latent_original_window);
            printf("val_logPriorLat:   %f\n", val_logPriorLat);

            printf("dim(Ax_window) = %ld x %ld\n", Ax_window.rows(), Ax_window.cols());
            printf("dim(mean_latent_original_window) = %ld\n", mean_latent_original_window.size());
            Vect eta_subset = Ax_window * mean_latent_original_window;
            std::cout << "eta(1:10) = " << eta_subset.head(10).transpose() << std::endl;
            
            SpMat Qxy(n_subset,n_subset);
            double log_det;

            //Vect mu = mean_latent_original + Vect::Random(n);
            Vect mu = mu_initial_window;
            std::cout << "initial  x : " << mu.head(min(10, (int) n_subset)).transpose() << std::endl;
            std::cout << "norm(init. lat - orig lat) : " << (mu - mean_latent_original_window).norm() << std::endl;

            SpMat Qx(n_subset,n_subset);
            fun->get_Qprior(theta_original, Qx); // construct_Qprior() will throw error, probably sth to do with Qx being internal variable?

            //string fileName_Qprior = "Qx_n" + to_string(n) + "_ns" + to_string(ns) + "_nt" + to_string(nt) + "_nss" + to_string(nss) + "_nb" + to_string(nb) + "_no" + to_string(no) + ".dat";
            //write_sym_CSC_matrix(fileName_Qprior, Qx);
            //std::cout << "Qx(1:10,1:10) = \n" << Qx.block(0,0,10,10) << std::endl;
            fun->NewtonIter(theta, mu, Qxy, log_det);
        
            /*string fileName_Q = "Qxy_n" + to_string(n) + "_ns" + to_string(ns) + "_nt" + to_string(nt) + "_nss" + to_string(nss) + "_nb" + to_string(nb) + "_no" + to_string(no) + ".dat";
            write_sym_CSC_matrix(fileName_Q, Qxy);
            std::cout << "Qxy(1:10,1:10) = \n" << Qxy.block(0,0,10,10) << std::endl;*/

            std::cout << "estimated x : " << mu.head(min(10, (int) n_subset)).transpose() << std::endl;
            std::cout << "original  x : " << mean_latent_original_window.head(min(10, (int) n_subset)).transpose() << "\n" << std::endl;

            std::cout << "estimated fixed effects : " << mu.tail(nb).transpose() << std::endl;
            std::cout << "original  fixed effects : " << mean_latent_original_window.tail(nb).transpose() << std::endl;
            std::cout << "norm(est. lat - orig lat) : " << (mu - mean_latent_original_window).norm() << std::endl;

            Vect marg(n_subset);
            fun->get_marginals_f(theta_original, mean_latent_original_window, marg);
            if(MPI_rank == 0){
                std::cout << "\nsd fixed effects:  " << marg.tail(nb).cwiseSqrt().transpose() << std::endl;
                std::cout << "sd random effects: " << marg.head(min(10,(int) n_subset)).cwiseSqrt().transpose() << std::endl;

            }
        } // end testing inner iteration

        MPI_Barrier(MPI_COMM_WORLD);

        double fx;

        double time_bfgs = 0.0;

        if(dim_th > 0){
            if(MPI_rank == 0){
                printf("\n====================== CALL BFGS SOLVER =====================\n");
            }

            //theta_param << -2.15, 9.57, 11.83, 3.24;
            fun->convert_interpret2theta(theta_param, theta);

            if(MPI_rank == 0){    
                std::cout << "theta param : " << theta_param.transpose() << std::endl;
                std::cout << "theta       : " << theta.transpose() << std::endl;
            }

            time_bfgs = -omp_get_wtime();
            int niter = solver.minimize(*fun, theta, fx, MPI_rank);
            //int niter = solver.minimize(*fun, theta_test.flat, fx, MPI_rank);

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

            if(MPI_rank == 0){
                fun->convert_interpret2theta(theta_original_param, theta_original);
                std::cout << "\norig. mean parameters        : " << theta_original.transpose() << std::endl;
                std::cout << "est.  mean parameters        : " << theta.transpose() << std::endl;
            }

            if(MPI_rank == 0){
                fun->convert_theta2interpret(theta_original, theta_original_param);
                std::cout << "\norig. mean interpret. param. : " << theta_original_param.transpose() << std::endl;

                fun->convert_theta2interpret(theta, theta_param);
                std::cout << "est.  mean interpret. param. : " << theta_param.transpose() << std::endl;
            }
        
            } else {
                if(MPI_rank == 0){
                    printf("\n====================== NO HYPERPARAMETERS JUST INNER ITERATION =====================\n");
                }

            SpMat Q(n,n);
            double log_det;
            Vect mu = Vect::Random(n);

            fun->NewtonIter(theta, mu, Q, log_det);
            if(MPI_rank == 0){
                std::cout << "mean fixed effects : " << mu.transpose() << std::endl;
                //std::cout << "logDet = " << log_det << std::endl;
            }

            Vect marg(n);
            fun->get_marginals_f(theta, mu, marg);
            if(MPI_rank == 0){
                std::cout << "\nsd fixed effects: " << marg.cwiseSqrt().transpose() << std::endl;
            }

            exit(1);

        }

        delete fun;

    }  // end loop over moving windows

    MPI_Finalize();
    return 0;
    
}
