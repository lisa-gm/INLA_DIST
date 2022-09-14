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
//#define RGF_SOLVER

#ifdef RGF_SOLVER
#include "cuda_runtime_api.h" // to use cudaGetDeviceCount()
#endif

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

    int threads_level1 = omp_get_max_threads();
    int threads_level2;

    #pragma omp parallel
    {  
    threads_level2 = omp_get_max_threads();
    }

    // overwrite in case RGF is used
    int noGPUs;
   
    if(MPI_rank == 0){
        printf("\n============== PARALLELISM & NUMERICAL SOLVERS ==============\n");
        printf("total no MPI ranks  : %d\n", MPI_size);
        printf("OMP threads level 1 : %d\n", threads_level1);
        printf("OMP threads level 2 : %d\n", threads_level2);
#ifdef RGF_SOLVER
	cudaGetDeviceCount(&noGPUs);
	printf("available GPUs      : %d\n", noGPUs);
#else
	printf("RGF dummy version\n");
    noGPUs = 0;
#endif
    }  
    
    if(argc != 1 + 6 && MPI_rank == 0){
        std::cout << "wrong number of input parameters. " << std::endl;

        std::cerr << "INLA Call : ns nt nb no path/to/files" << std::endl;

        std::cerr << "[integer:ns]                number of spatial grid points " << std::endl;
        std::cerr << "[integer:nt]                number of temporal grid points " << std::endl;
        std::cerr << "[integer:nb]                number of fixed effects" << std::endl;
        std::cerr << "[integer:no]                number of data samples" << std::endl;

        std::cerr << "[string:base_path]          path to folder containing matrix files " << std::endl;

        exit(1);
    }

#ifdef PRINT_MSG
    if(MPI_rank == 0){
        std::cout << "reading in example. " << std::endl;
    }
#endif

    size_t ns = atoi(argv[1]);
    size_t nt = atoi(argv[2]);
    size_t nb = atoi(argv[3]);
    size_t no = atoi(argv[4]);

    // to be filled later

    // set nt = 1 if ns > 0 & nt = 0
    if(ns > 0 && nt == 0){
        nt = 1;
    } 

    size_t n = ns*nt + nb;


    // also save as string
    std::string ns_s = std::to_string(ns);
    std::string nt_s = std::to_string(nt);
    std::string nb_s = std::to_string(nb);
    std::string no_s = std::to_string(no); 
    std::string n_s  = std::to_string(ns*nt + nb);

    std::string base_path = argv[5];    

    std::string solver_type = argv[6];

    // check if solver type is neither PARDISO nor RGF :
    if(solver_type.compare("PARDISO") != 0 && solver_type.compare("RGF") != 0){
        std::cout << "Unknown solver type. Available options are :\nPARDISO\nRGF" << std::endl;
        exit(1);
    }

    if(MPI_rank == 0){
        std::cout << "Solver : " << solver_type << std::endl;
    }

    // we have two cholesky factors ...
    if(MPI_rank == 0 && solver_type.compare("RGF") == 0){
        // required memory on CPU to store Cholesky factor
        double mem_gb = (2*(nt-1)*ns*ns + ns*ns + (ns*nt+nb)*nb) * sizeof(double) / pow(10.0,9.0);
        printf("Memory Usage of each Cholesky factor on CPU = %f GB\n\n", mem_gb);
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

    int num_constr = 1;
    bool constr = false;
    //bool constr = true;
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

        /*if(MPI_rank == 0){
            std::cout << "total number of observations : " << no << std::endl;
        }*/

    } else {
        if(MPI_rank == 0){
            std::cout << "invalid parameters : ns nt !!" << std::endl;
            exit(1);
        }    
    }


//#ifdef DATA_TEMPERATURE
    // data y
    std::string y_file        =  base_path + "/y_" + no_s + "_1" + ".dat";
    file_exists(y_file);
    // at this point no is set ... 
    // not a pretty solution. 
    y = read_matrix(y_file, no, 1);  
    if(MPI_rank == 0) 
        std::cout << "sum(y) = " << y.sum() << std::endl;

//#endif


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


    /* ----------------------- initialise random theta -------------------------------- */

    Vect theta(dim_th);             // define initial guess in model parametrization
    Vect theta_param(dim_th);       // or in interpretable parametrization
    Vect theta_prior_param(dim_th);
    Vect theta_original(dim_th); theta_original.setZero();

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
        // sigma.e (noise observations), sigma.u, range s, range t
        //theta_original_param << 0.5, 4, 1, 10;
        // sigma.e (noise observations), gamma_E, gamma_s, gamma_t
        theta_original << 1.386294, -5.882541,  1.039721,  3.688879;  // here exact solution, here sigma.u = 4
        //theta_prior << 1.386294, -5.594859,  1.039721,  3.688879; // here sigma.u = 3
        //theta_prior << 1.386294, -5.594859, 1.039721,  3.688879; // here sigma.u = 3
        // using PC prior, choose lambda  
        theta_prior_param << 0.7/3.0, 0.2*0.7*0.7, 0.7, 0.7/3.0;

        //theta_param << 1.373900, 2.401475, 0.046548, 1.423546; 
        //theta << 1, -3, 1, 3;   // -> the one used so far !! maybe a bit too close ... 
        theta_param << 4, 0, 0, 0;
        //theta_param << 4,4,4,4;
        //theta_param << 1.366087, 2.350673, 0.030923, 1.405511;
        /*theta << 2, -3, 1.5, 5;
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

        if(MPI_rank == 0){
            std::cout << "using TEMPERATURE DATASET" << std::endl; 
            if(constr)
                std::cout << "assuming sum-to-zero constraints on spatial-temporal field." << std::endl;
        }
        //theta << 4, 4, 4, 4;    //
        //theta_param << 4, 0, 0, 0;
        //theta_param << -1.308664,  0.498426,  4.776162,  1.451209;
        theta_param << -1.270, 12.132, 9.773, 4.710;
        //theta_param << -1.25, 13.6, 10.4, 5.4;
        theta_original << -1.269613,  5.424197, -8.734293, -6.026165;  // estimated from INLA / same for my code varies a bit according to problem size


        // using PC prior, choose lambda  
        theta_prior_param << 0.7/3.0, 0.2*0.7*0.7, 0.7, 0.7/3.0;

        //std::cout << "theta prior        : " << std::right << std::fixed << theta_prior.transpose() << std::endl;
        //theta << -0.2, -2, -2, 3;
        /*if(MPI_rank == 0){
            std::cout << "initial theta      : "  << std::right << std::fixed << theta.transpose() << std::endl;
        }*/

        if(constr){

#if 1
            // =============== 1 SUM-TO-ZERO CONSTRAINT PER K TIME-STEPS ==================== //
            // number of time-steps per constraint 
            int tsPerConstr = 1;
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
    // or if objective function has not decreased by more than  
    // cant find epsilon_rel in documentation ...
    // stops if grad.norm() < eps_rel*x.norm() 
    param.epsilon_rel=1e-3;
    //param.epsilon_rel = 1e-5;
    // in the past ... steps
    param.past = 2;
    // TODO: stepsize too small? seems like it almost always accepts step first step.    
    // changed BFGS convergence criterion, now stopping when abs(f(x_k) - f(x_k-1)) < delta
    // is this sufficiently bullet proof?!
    param.delta = 1e-3;
    //param.delta = 1e-10;
    // maximum line search iterations
    param.max_iterations = 200;

    // Create solver and function object
    LBFGSSolver<double> solver(param);

    /*std::cout << "\nspatial grid size  : " << std::right << std::fixed << g1.rows() << " " << g1.cols() << std::endl;
    std::cout << "temporal grid size : " << M1.rows() << " " << M1.cols() << std::endl;
    std::cout << "Ax size            : " << Ax.rows() << " " << Ax.cols() << std::endl;*/

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
        fun = new PostTheta(ns, nt, nb, no, Ax, y, c0, g1, g2, theta_prior_param, solver_type, constr, Dx, Dxy, validate, w);
    } else {
        if(MPI_rank == 0){
            std::cout << "\ncall spatial-temporal constructor." << std::endl;
        }
        fun = new PostTheta(ns, nt, nb, no, Ax, y, c0, g1, g2, g3, M0, M1, M2, theta_prior_param, solver_type, constr, Dx, Dxy, validate, w);
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
            std::cout << "initial theta      : "  << std::right << std::fixed << theta.transpose() << std::endl;
            std::cout << "initial theta interpret. param. : " << theta_interpret_initial.transpose() << std::endl;
        }
    }



    double fx;

#if 0
	double t_f_eval = -omp_get_wtime();

    ArrayXi fact_to_rank_list(2);
    fact_to_rank_list << 0,0;
    if(MPI_size >= 2){
        fact_to_rank_list[1] = 1; 
        }
    std::cout << "i = " << 0 << ", MPI rank = " << MPI_rank << ", fact_to_rank_list = " << fact_to_rank_list.transpose() << std::endl;
            
    if(MPI_rank == fact_to_rank_list[0] || MPI_rank == fact_to_rank_list[1]){

    	// single function evaluation
    	for(int i=0; i<1; i++){

    		Vect mu_dummy(n);
    		fx = fun->eval_post_theta(theta_original, mu_dummy, fact_to_rank_list);
            //fx = fun->eval_post_theta(theta_original, mu_dummy);

    		std::cout <<  "f(x) = " << fx << std::endl;

        }
    }

	t_f_eval += omp_get_wtime();
	std::cout << "time in f eval loop : " << t_f_eval << std::endl;

#endif

#if 0
    if(MPI_rank == 0)
        printf("\n====================== CALL BFGS SOLVER =====================\n");

    //LIKWID_MARKER_INIT;
    //LIKWID_MARKER_THREADINIT;

    double time_bfgs = -omp_get_wtime();
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
        double prior_sigU; double prior_ranS; double prior_ranT;
        fun->convert_theta2interpret(theta_original[1], theta_original[2], theta_original[3], prior_ranT, prior_ranS, prior_sigU);
        std::cout << "\norig. mean interpret. param. : " << theta_original[0] << " " << prior_ranT << " " << prior_ranS << " " << prior_sigU << std::endl;

        double lgamE = theta[1]; double lgamS = theta[2]; double lgamT = theta[3];
        double sigU; double ranS; double ranT;
        fun->convert_theta2interpret(lgamE, lgamS, lgamT, ranT, ranS, sigU);
        std::cout << "est.  mean interpret. param. : " << theta[0] << " " << ranT << " " << ranS << " " << sigU << std::endl;
    }

    #endif

#if 0
    Vect theta_max(dim_th);
    //theta_max << 2.675054, -2.970111, 1.537331;    // theta
    //theta_max = theta_prior;
    theta_max = theta_original;

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
   
#ifdef PRINT_MSG 
    if(MPI_rank == 0){
        std::cout << "est.  mean interpret. param. : " << interpret_theta[0] << " " << interpret_theta[1] << " " << interpret_theta[2] << " " << interpret_theta[3] << std::endl;
    }
#endif

    //double t_get_covariance = -omp_get_wtime();
    t_get_covariance = -omp_get_wtime();
    cov = fun->get_Cov_interpret_param(interpret_theta, eps);
    t_get_covariance += omp_get_wtime();


    if(MPI_rank == 0){
        std::cout << "\ncovariance interpr. param.  : \n" << cov << std::endl;
        //std::cout << "time get covariance         : " << t_get_covariance << " sec" << std::endl;
    }
    #endif


#if 1
    double t_get_fixed_eff;
    Vect mu(n);

    ArrayXi fact_to_rank_list(2);
    fact_to_rank_list << 0,0;
    if(MPI_size >= 2){
        fact_to_rank_list[1] = 1; 
    }

    if(MPI_rank == fact_to_rank_list[0] || MPI_rank == fact_to_rank_list[1]){
        std::cout << "MPI rank = " << MPI_rank << ", fact_to_rank_list = " << fact_to_rank_list.transpose() << std::endl;

        t_get_fixed_eff = - omp_get_wtime();
        fun->get_mu(theta_original, mu, fact_to_rank_list);
        t_get_fixed_eff += omp_get_wtime();
    }

    // CAREFUL! at the moment mu is in rank 1 ... how to do this properly??
    if(MPI_rank == fact_to_rank_list[1]){

        std::cout << "\nestimated mean fixed effects : " << mu.tail(nb).transpose() << std::endl;
        std::cout << "estimated mean random effects: " << mu.head(10).transpose() << std::endl;
        //std::cout << "time get fixed effects       : " << t_get_fixed_eff << " sec\n" << std::endl;

#ifdef PRINT_MSG
        if(constr == true)
            std::cout << "Dxy*mu : " << (Dxy*mu).transpose() << ", \nshould be : " << e.transpose() << std::endl;
#endif
    }


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
#endif

    
#endif


  
    // =================================== compute marginal variances =================================== //
#if 1
    double t_get_marginals;
    Vect marg(n);

    // when the range of u is large the variance of b0 is large.
    if(MPI_rank == 0){
        std::cout << "\n==================== compute marginal variances ================" << std::endl;
        //theta << -1.269613,  5.424197, -8.734293, -6.026165; // most common solution for temperature dataset
        std::cout << "\nUSING ESTIMATED THETA : " << theta_original.transpose() << std::endl;
        
        t_get_marginals = -omp_get_wtime();
        fun->get_marginals_f(theta_original, marg);
        t_get_marginals += omp_get_wtime();

        //std::cout << "\nest. variances fixed eff.    :  " << marg.tail(10).transpose() << std::endl;
        std::cout << "est. standard dev fixed eff  : " << marg.tail(nb).cwiseSqrt().transpose() << std::endl;
        std::cout << "est. std dev random eff      : " << marg.head(10).cwiseSqrt().transpose() << std::endl;
        //std::cout << "diag(Cov) :                     " << Cov.diagonal().transpose() << std::endl;
    }
/*
#ifdef DATA_SYNTHETIC
        // ============================================ //
        Vect marg_original(n);
        fun->get_marginals_f(theta_original, marg_original);

        std::cout << "\nUSING ORIGINAL THETA : " << theta_original.transpose() << std::endl;
        //std::cout << "\nest. variances fixed eff.    :  " << marg.tail(10).transpose() << std::endl;
        std::cout << "est. standard dev fixed eff  : " << marg_original.tail(nb).cwiseSqrt().transpose() << std::endl;
        std::cout << "est. std dev random eff      : " << marg_original.head(10).cwiseSqrt().transpose() << std::endl;
        //std::cout << "diag(Cov) :                     " << Cov.diagonal().transpose() << std::endl;
#endif

        // ============================================ //
        Vect theta_INLA(dim_th);
#ifdef DATA_SYNTHETIC
        if(no == 128400)
            theta_INLA << 1.384902, -5.880753, 1.015273, 3.709717;  // 642, 20, 128400
        if(no == 25680)
            theta_INLA << 1.393959, -5.915735, 1.262288, 3.475778;    // 642, 20,  25680
        if(no == 2520)
            theta_INLA << 1.306129, -5.840268, 0.879396, 3.718468;
        if(no == 630)
             theta_INLA <<  2.005811, -6.074556, 1.112258,  3.828027;
#elif defined(DATA_TEMPERATURE)
        Vect theta_INLA_param(4);
        theta_INLA_param << -1.270, 12.132, 9.773, 4.710;
        //theta_INLA_param << -1.352, 1.912, 3.301, 3.051;

        theta_INLA[0] = theta_INLA_param[0];
        fun->convert_interpret2theta(theta_INLA_param[1], theta_INLA_param[2], theta_INLA_param[3], theta_INLA[1], theta_INLA[2], theta_INLA[3]);

#endif

        Vect marg_INLA(n);
        fun->get_marginals_f(theta_INLA, marg_INLA);

        std::cout << "\nUSING THETA FROM INLA : " << theta_INLA.transpose() << std::endl;
        //std::cout << "\nest. variances fixed eff.    :  " << marg.tail(10).transpose() << std::endl;
        std::cout << "est. standard dev fixed eff  : " << marg_INLA.tail(nb).cwiseSqrt().transpose() << std::endl;
        std::cout << "est. std dev random eff      : " << marg_INLA.head(10).cwiseSqrt().transpose() << std::endl;
        //std::cout << "diag(Cov) :                     " << Cov.diagonal().transpose() << std::endl;
   */

#endif  // #if 0/1 for marginals


#if 0
    // =================== expected marginal variances ============================ //
    // first way
    // construct Q = Qx(theta_mode) + theta_mode*AxTAx => assume Q_b = 1e-5*Id
    // compute inverse -> using pardiso & using .inverse() => extract diagonal => compare

    SpMat Qst(ns*nt,ns*nt);  // not n?!
    construct_Q_spat_temp(theta, c0, g1, g2, g3, M0, M1, M2, Qst);

    //SpMat epsId()

    SpMat Qx(n,n);        
    int nnz = Qst.nonZeros();
    Qx.reserve(nnz);

    for (int k=0; k<Qst.outerSize(); ++k){
      for (SparseMatrix<double>::InnerIterator it(Qst,k); it; ++it)
      {
        Qx.insert(it.row(),it.col()) = it.value();                 
      }
    }

    for(int i=ns*nt; i < n; i++){
        Qx.coeffRef(i,i) = 1e-3;
    }

    Qx.makeCompressed();

    SpMat Q =  Qx + exp(theta[0]) * Ax.transpose()*Ax;


    if(MPI_rank == 0){
        //std::cout << "Qx(0,0,10,10) : \n" << Qx.block(0,0,10,10) << std::endl;
        //std::cout << "Ax(0,0,30,30) : \n" << Ax.block(0,0,30,30) << std::endl;
        
        //MatrixXd AxTAx = exp(theta[0]) * Ax.transpose()*Ax;
        //std::cout << "exp(theta[0]) = " << exp(theta[0]) << std::endl;
        //std::cout << "AxTAx.topLeft(30,30) : \n" << AxTAx.topLeftCorner(30,30) << std::endl;
        //std::cout << "AxTAx.bottomRight(10,10) : \n" << AxTAx.bottomRightCorner(10,10) << std::endl;


        MatrixXd Q_d = MatrixXd(Q);
        std::cout << "Q_topLeft(0,0,10,10) : \n" << Q_d.topLeftCorner(10,10) << std::endl;
        std::cout << "Q_bottomRight(0,0,10,10) : \n" << Q_d.bottomRightCorner(10,10) << std::endl;

        /*MatrixXd invQ_d = Q_d.inverse();
        std::cout << "\nmarginals fixed effects = " << invQ_d.diagonal().tail(10).transpose() << std::endl;
        std::cout << "marginals random effects= " << invQ_d.diagonal().head(10).transpose() << std::endl;*/

        Solver* solverQ;
        solverQ   = new PardisoSolver(MPI_rank);

        Vect inv_diag(n);
        solverQ->selected_inversion(Q, inv_diag);
        std::cout << "\nmarg std dev FE 'true'       : " << inv_diag.tail(10).cwiseSqrt().transpose() << std::endl;
        std::cout << "marg std dev RE 'true'       : " << inv_diag.head(10).cwiseSqrt().transpose() << std::endl;

        // invert using eigen
        /*SimplicialLDLT<SparseMatrix<double>> solverEigen;
        solverEigen.compute(Q);

        SpMat I(n,n); I.setIdentity();
        MatrixXd fullInv(n,n);

        std::cout << "identity matrix : \n" << I.block(0,0,10,10) << std::endl;

        fullInv = solverEigen.solve(I);
        std::cout << "norm(I - Q*fullInv) = " << (I - Q*fullInv).norm() << std::endl;

        std::cout << "marg std dev FE fullInv      : " << fullInv.diagonal().tail(10).cwiseSqrt().transpose() << std::endl;
        std::cout << "marg std dev RE fullInv      : " << fullInv.diagonal().head(10).cwiseSqrt().transpose() << std::endl;
        */

    }    
#endif

    // =================================== print times =================================== //
#if 0
    t_total +=omp_get_wtime();
    if(MPI_rank == 0){
        // total number of post_theta_eval() calls 
        //std::cout << "\ntotal number fn calls        : " << fun->get_fct_count() << std::endl;
        std::cout << "\ntime BFGS solver             : " << time_bfgs << " sec" << std::endl;
        std::cout << "time get covariance          : " << t_get_covariance << " sec" << std::endl;
        std::cout << "time get marginals FE        : " << t_get_marginals << " sec" << std::endl;
        std::cout << "total time                   : " << t_total << " sec" << std::endl;
    }
    #endif


    // ======================== write LOG file ===================== //
    // mpi_size, no_threads level 1, level 2, solver_type
    // ns, nt, nb, no, synthetic/real, theta_prior, theta_original (set all zero in real case), in both 
    // parametrisations?
    // theta_max, interpret_param_theta_max
    // mean fixed effects
    // var/std fixed effects
    // no of iterations BFGS
    // runtimes: BFGS, time per iteration, covariance (hessian), partial inversion, total

#ifdef WRITE_LOG
    if(MPI_rank == 0){

        // add time stamp to not overwrite?
        std::string log_file_name = base_path + "/log_"+ solver_type + "_ns" + ns_s + "_nt" + nt_s + "_nb" + nb_s + "_no" + no_s +".dat";
        std::ofstream log_file(log_file_name);
        log_file << "mpi_size\t" << MPI_size << std::endl;
        log_file << "threads_l1\t" << threads_level1 << std::endl;
        log_file << "threads_l2\t" << threads_level2 << std::endl;
        log_file << "noGPUs\t" << noGPUs << std::endl;
        log_file << "solver_type\t" << solver_type << std::endl;
        log_file << "datatype\t" << data_type << std::endl;
        log_file << "ns\t" << ns << std::endl;
        log_file << "nt\t" << nt << std::endl;
        log_file << "num_fixed\t" << nb << std::endl;
        log_file << "num_obs\t" << no << std::endl;
        log_file << "theta_prior_param\t" << theta_prior_param.transpose() << std::endl;
        log_file << "theta_original\t" << theta_original.transpose() << std::endl;
        log_file << "theta_max\t" << theta_max.transpose() << std::endl;
        log_file << "interpret_param_theta_max\t" << interpret_theta.transpose() << std::endl;
	log_file << "mean_fixed_eff\t" << mu.tail(nb).transpose() << std::endl;
        log_file << "std_dev_fixed_eff\t" << marg.tail(nb).cwiseSqrt().transpose() << std::endl;          log_file << "time_bfgs\t" << time_bfgs << std::endl;
        log_file << "noIter\t" << niter << std::endl; 
        log_file << "time_Iter\t" << time_bfgs/niter << std::endl;
        log_file << "time_Cov\t" << t_get_covariance << std::endl;               
        log_file << "time_MargFE\t" << t_get_marginals <<  std::endl;
        log_file << "t_total\t" << t_total << std::endl;

  log_file.close(); 

    }

#endif


    delete fun;

    MPI_Finalize();
    return 0;
    
}
