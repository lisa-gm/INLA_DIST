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

// std::setwd print out
#include <iomanip>

#include <Eigen/Dense>
#include <Eigen/CholmodSupport>
#include <unsupported/Eigen/KroneckerProduct>

//#include "solver_cholmod.h" -> pardiso can do inversion now
#include "PardisoSolver.h"
#include "RGFSolver.h"

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

	Solver** solverQ;
	Solver** solverQst;

	string solver_type;

	//PardisoSolver* solverQ;   /**<  list of Pardiso solvers, for denom.		*/
	//PardisoSolver* solverQst; /**<  list of Pardiso solvers, for Qu         */

	int fct_count;      /**< count total number of function evaluations 	*/

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

	public:
	 /**
     * @brief constructor for regression model (no random effects). 
     * @param[in] ns_ number of spatial grid points per time step.
     * @param[in] nt_ number of temporal time steps.
     * @param[in] nb_ number of fixed effects.
     * @param[in] no_ number of observations.
     * @param[in] B_  covariate matrix.
     * @param[in] y_  vector with observations.
     * \note B = B_ or is its own copy?
     */	
	PostTheta(int ns, int nt, int nb, int no, MatrixXd B, VectorXd y, Vector theta_prior, string solver_type);
	/**
     * @brief constructor for spatial model (order 2).
     * @param[in] ns_ number of spatial grid points per time step.
     * @param[in] nt_ number of temporal time steps.
     * @param[in] nb_ number of fixed effects.
     * @param[in] no_ number of observations.
     * @param[in] Ax_  covariate matrix.
     * @param[in] y_  vector with observations.
     * @param[in] c0_ diagonalised mass matrix.
     * @param[in] g1_ stiffness matrix.
     * @param[in] g2_ defined as : g1 * c0^-1 * g1
     */	
	PostTheta(int ns, int nt, int nb, int no, SpMat Ax, VectorXd y, SpMat c0, SpMat g1, SpMat g2, Vector theta_prior, string solver_type);

	/**
     * @brief constructor for spatial temporal model.
     * @brief constructor for spatial model (order 2).
     * @param[in] ns_ number of spatial grid points per time step.
     * @param[in] nt_ number of temporal time steps.
     * @param[in] nb_ number of fixed effects.
     * @param[in] no_ number of observations.
     * @param[in] Ax_  covariate matrix.
     * @param[in] y_  vector with observations.
     * @param[in] c0_ diagonalised mass matrix space.
     * @param[in] g1_ stiffness matrix space.
     * @param[in] g2_ defined as : g1 * c0^-1 * g1
     * @param[in] g3_ defined as : g1 * (c0^-1 * g1)^2
     * @param[in] M0_ diagonalised mass matrix time.
     * @param[in] M1_ diagonal matrix with diag(0.5, 0, ..., 0, 0.5) -> account for boundary
     * @param[in] M2_ stiffness matrix time.
     */	
	PostTheta(int ns, int nt, int nb, int no, SpMat Ax, VectorXd y, SpMat c0, SpMat g1, SpMat g2, SpMat g3, SpMat M0, SpMat M1, SpMat M2, Vector theta_prior, string solver_type); 

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
     * @brief check if Hessian positive definite (matrix assumed to be dense & small since dim(theta) small)
 	 * @param[inout] updates hessian to only the diagonal entries if not positive definite.
     */
     void check_pos_def(MatrixXd &hess);

	// ============================================================================================ //
	// ALL FOLLOWING FUNCTIONS CONTRIBUT TO THE EVALUATION OF F(THETA) & GRADIENT

	/**
     * @brief Core function. Evaluate posterior of theta. mu are latent parameters.
     * @param[in]    theta hyperparameter vector
 	 * @param[inout] mu vector of the conditional mean
 	 * @return 		 f(theta) value
     */
	double eval_post_theta(Vector& theta, Vector& mu);

	/**
     * @brief evaluate log prior using original theta value
     * @param[in] thetai current theta_i value
     * @param[in] thetai_original original theta_i value
 	 * @param[inout] log prior is being updated.
 	 * @details variance / precision of 1 : no normalising constant. 
 	 * computed through -0.5 * (theta_i* - theta_i)*(theta_i*-theta_i) 
     */	
	void eval_log_prior(double& log_prior, double* thetai, double* thetai_original);

	/**
	 * @brief evaluate log prior of random effects
	 * @param[in] theta current theta vector
		 * @param[inout] log_det inserts log determinant.
		 * \todo construct spatial matrix (at the moment this is happening twice. FIX)
	 */	
	void eval_log_det_Qu(Vector& theta, double &log_det);

	/**
     * @brief compute log likelihood : log_det tau*no and value -theta*yTy
     * @param[in] theta current theta vector
 	 * @param[inout] log_det inserts log determinant of log likelihood.
 	 * @param[inout] val inserts the value of -theta*yTy
     */	
	void eval_likelihood(Vector& theta, double &log_det, double &val);
	
	/**
     * @brief spatial model : SPDE discretisation -- matrix construction
     * @param[in] theta current theta vector
 	 * @param[inout] Qs fills spatial precision matrix
     */
	void construct_Q_spatial(Vector& theta, SpMat& Qs);

	/**
     * @brief spatial temporal model : SPDE discretisation. DEMF(1,2,1) model.
     * @param[in] theta current theta vector
 	 * @param[inout] Qst fills spatial-temporal precision matrix
     */
	void construct_Q_spat_temp(Vector& theta, SpMat& Qst);

	/** @brief construct precision matrix. 
	 * Calls spatial, spatial-temporal, etc.
     * @param[in] theta current theta vector
 	 * @param[inout] Q fills precision matrix
     */
	void construct_Q(Vector& theta, SpMat& Q);

	/** @brief Assemble right-handside. 
     * @param[in] theta current theta vector
 	 * @param[inout] rhs right-handside
 	 * /todo Could compute Ax^T*y once, and only multiply with appropriate exp_theta.
     */	
	void construct_b(Vector& theta, Vector &rhs);

	/** @brief Evaluate denominator: conditional probability of Qx|y
     * @param[in] theta current theta vector
     * @param[inout] log_det fill log determinant of conditional distribution of denominator
     * @param[inout] val fill value with mu*Q*mu
     * @param[inout] Q construct precision matrix
 	 * @param[inout] rhs construct right-handside
 	 * @param[inout] mu insert mean of latent parameters
     */
	void eval_denominator(Vector& theta, double& log_det, double& val, SpMat& Q, Vector& rhs, Vector& mu);

	// ============================================================================================ //
	// FINITE DIFFERENCE GRADIENT EVALUATION

	/** @brief Compute gradient using central finite difference stencil. Parallelised with OpenMP 
     * @param[in] theta current theta vector
 	 * @param[in] f_theta value of f(theta) 
 	 * @param[mu] mu 
 	 * @param[inout] grad inserts gradient 
 	 * \todo don't actually need gradient?
     */
	void eval_gradient(Vector& theta, double f_theta, Vector& mu, Vector& grad);

	 /**
     * @brief class destructor. Frees memory allocated by PostTheta class.
     */
	~PostTheta();

};

#endif
