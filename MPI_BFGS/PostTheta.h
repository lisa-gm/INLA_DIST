#ifndef POST_THETA_H
#define POST_THETA_H

#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include <omp.h>
#include "mpi.h"


// std::setwd print out
#include <iomanip>

#include <Eigen/Dense>
#include <Eigen/CholmodSupport>
#include <unsupported/Eigen/KroneckerProduct>

//#include "solver_cholmod.h" -> pardiso can do inversion now
#include "PardisoSolver.h"
#include "RGFSolver.h"

#include "Model.h"

//#define PRINT_MSG
//#define PRINT_TIMES

using namespace Eigen;
using namespace std;

// typedef Eigen::Matrix<double, Dynamic, Dynamic> Mat;
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::VectorXd Vector;


 /**
 * @brief Computes the Posterior of the hyperparameters theta. 
 * @details Computes the posterior of theta for a given theta and its gradient using a central finite difference approximation. Can additionally compute an approximation to the Hessian. 
 */
class PostTheta{

	private:

	int ns;				/**<  number of spatial grid points per timestep 	*/
	int nt;				/**<  number of temporal time steps 				*/
    int nb;				/**<  number of fixed effects 						*/
    int no;				/**<  number of observations 						*/
    int nu;				/**<  number of random effects, that ns*nu 			*/
    int n;				/**<  total number of unknowns, i.e. ns*nt + nb 	*/

	int dim_th;			/**<  dimension of hyperparameter vector theta 		*/
	int threads_level1; /**<  number of threads on first level 				*/
	int dim_grad_loop;  /**<  dimension of gradient loop 					*/
	int num_solvers;    /**<  number of pardiso solvers 					*/

	string solver_type;

	//PardisoSolver* solverQ;   /**<  list of Pardiso solvers, for denom.		*/
	//PardisoSolver* solverQst; /**<  list of Pardiso solvers, for Qu         */

	int fct_count;      /**< count total number of function evaluations 	*/
	int iter_count;		/**< count total number of operator() call        	*/

    VectorXd y; 		/**<  vector of observations y. has length no. 		*/
    Vector theta_prior; /**<  vector with prior values. Constructs normal
 						      distribution with sd = 1 around these values. */

    // either Ax or B used
    SpMat Ax;			/**< sparse matrix of size no x (nu+nb). Projects 
    						 observation locations onto FEM mesh and 
    						 includes covariates at the end.                */
    MatrixXd B; 		/**< if space (-time) model included in last 
    						 columns of Ax. For regression only B exists.   */

    // used in spatial and spatial-temporal case
    SpMat c0;			/**< Diagonal mass matrix spatial part. 			*/
    SpMat g1;			/**< stiffness matrix space. 						*/
    SpMat g2;			/**< defined as : g1 * c0^-1 * g1  					*/

    // only used in spatial-temporal case
    SpMat g3;			/**< defined as : g1 * (c0^-1 * g1)^2.				*/
    SpMat M0;			/**< diagonalised mass matrix time. 				*/
    SpMat M1;			/**< diagonal matrix with diag(0.5, 0, ..., 0, 0.5) 
    							-> account for boundary						*/
    SpMat M2;			/**< stiffness matrix time.							*/

    double yTy;			/**< compute t(y)*y once. */
    Vector mu;			/**< conditional mean */
    Vector t_grad;		/**< gradient of theta */
    double min_f_theta; /**< minimum of function*/
	double f_theta;


    double* theta_array;
    int no_f_eval;

	public:
	/**
     * @brief constructor for PostTheta class using MPI.
     * @description determines the gradient and function value at theta using a central finite difference approximation.
     * Individual function evaluations are sent via MPI_Send/Recv to workers (MPI processes with rank > 0), where the models
     * are constructed individually using the appropriate theta. 
     * @param[in] dim_th_ dimension of hyperparameter vector theta.
     */	
	PostTheta(int dim_th_); 

	/**
     * @brief structure required by BFGS solver, requires : theta, gradient theta
	 * \note Gradient call is already parallelised using nested OpenMP. 
	 * --> there are l1 threads (usually 8, one for each function evaluation), that themselves
	 * then split into another e.g. 8 threads, when calling PARDISO to factorise the system.
	 * --> somehow introduce additional parallelism to compute f(theta), possible to do in parallel
	 */
    double operator()(Vector& theta, Vector& grad);

    int get_fct_count();

	// ============================================================================================ //
	// CONVERT MODEL PARAMETRISATION TO INTERPRETABLE PARAMETRISATION & VICE VERSA

	/**
	 * @brief convert hyperparameters theta from the model parametrisation to the interpretable
	 * parametrisation ie. from log(gamma_E, gamma_s, gamma_t) to log(sigma.u, rangeS, rangeT)
	 * @param [in]		log(gamma_E)
	 * @param [in]		log(gamma_s)
	 * @param [in]		log(gamma_t)
	 * @param [inout]	log(sigma.u) precision of random effects
	 * @param [inout]	log(ranS) 	 spatial range
	 * @param [inout]	log(ranT) 	 temporal range
	 */
	void convert_theta2interpret(double lgamE, double lgamS, double lgamT, double& sigU, double& ranS, double& ranT);
	
	/**
	 * @brief convert hyperparameters theta from the interpretable parametrisation to the
	 * model parametrisation ie. from log(sigma.u, rangeS, rangeT) to log(gamma_E, gamma_s, gamma_t)
	 * @param [in]		log(sigma.u) precision of random effects
	 * @param [in]		log(ranS) 	 spatial range
	 * @param [in]		log(ranT) 	 temporal range
	 * @param [inout]	log(gamma_E) 
	 * @param [inout]	log(gamma_s)
	 * @param [inout]	log(gamma_t)
	 */
	void convert_interpret2theta(double sigU, double ranS, double ranT, double& lgamE, double& lgamS, double& lgamT);

	// ============================================================================================ //
	// FUNCTIONS TO BE CALLED AFTER THE BFGS SOLVER CONVERGED

	/**
     * @brief get conditional mean mu for theta.
     * @param [in]    theta hyperparameter vector
     * @param [inout] mu vector of the conditional mean
     */	
	void get_mu(Vector& theta, Vector& mu);

	/**
     * @brief returns current gradient of theta.
     * @return gradient_theta
     */	
	Vector get_grad();

	/**
     * @brief Compute Covariance matrix of hyperparameters theta, at theta.
     * @details computes the hessian of f(theta) using a second order finite
     * difference stencil and then inverts the hessian. Gaussian assumption.
     * @param [in] theta hyperparameter Vector
     * @return cov covariance matrix of the hyperparameters
     */	
	MatrixXd get_Covariance(Vector& theta);

	/**
     * @brief Compute the marginal variances of the latent parameters at theta. 
 	 * Using selected inversion procedure.
 	 * @param[in]  	 Vector theta.
 	 * @param[inout] Vector with marginals of f.
     */	
	void get_marginals_f(Vector& theta, Vector& vars);

	/**
     * @brief computes the hessian at theta using second order finite difference.
 	 * Is used be get_Covariance.
 	 * @param[in] theta hyperparameter vector
 	 * @return Dense Matrix with Hessian. 
 	 * \todo not yet parallelised .... 
     */	
	MatrixXd hess_eval(Vector& theta);

	/** 
	 * @brief scheduler for assigning work to workers
	 * @param[inout] counter
	 * @param[in] 	 mpi size (number of workers)
	 */
	void update_counter(int counter, int mpi_size);

	/**
     * @brief check if Hessian positive definite (matrix assumed to be dense & small since dim(theta) small)
 	 * @param[inout] updates hessian to only the diagonal entries if not positive definite.
     */
     void check_pos_def(MatrixXd &hess);

	 /**
     * @brief class destructor. Frees memory allocated by PostTheta class.
     */
	~PostTheta();

};

#endif
