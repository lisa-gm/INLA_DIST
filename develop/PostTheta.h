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
#include <Eigen/Core>
#include <Eigen/SparseCholesky>
//#include <Eigen/CholmodSupport>
#include <unsupported/Eigen/KroneckerProduct>

//#include "solver_cholmod.h" -> pardiso can do inversion now
#include "PardisoSolver.h"
#include "RGFSolver.h"
//#include "RGFSolver_dummy.h"
#include "EigenCholSolver.h"

//#include "Hyperparameters.h"

//#define SMART_GRAD

//#define PRINT_MSG
//#define PRINT_TIMES
//#define RECORD_TIMES
//#define DATA_SYNTHETIC

using namespace Eigen;
using namespace std;

// typedef Eigen::Matrix<double, Dynamic, Dynamic> Mat;
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::SparseMatrix<double, RowMajor> SpRmMat;
typedef Eigen::VectorXd Vect;


 /**
 * @brief Computes the Posterior of the hyperparameters theta. 
 * @details Computes the posterior of theta for a given theta and its gradient using a central finite difference approximation. Can additionally compute an approximation to the Hessian. 
 */
class PostTheta{

    private:

    int MPI_size;       /**< number of mpi ranks                            */
    int MPI_rank;       /**< personal mpi rank                              */
    int threads_level1; /**<  number of threads on first level              */
    int threads_level2;


    int ns;             /**<  number of spatial grid points per timestep    */
    int nt;             /**<  number of temporal time steps                 */
    int nss;            /**<  size of add. spatial field, not def = 0       */
    int nb;             /**<  number of fixed effects                       */
    int no;             /**<  number of observations                        */
    int nst;            /**<   ns*nt, equal to nu if nss =0                 */
    int nu;             /**<  number of random effects, that ns*nu          */
    int n;              /**<  total number of unknowns, i.e. ns*nt + nb     */

    size_t nnz_Qst;
    size_t nnz_Qs;

    int dim_th;         /**<  dimension of hyperparameter vector theta      */
    int dim_spatial_domain; 
    string manifold;    /**<  in R^d or on the sphere */
    int dim_grad_loop;  /**<  dimension of gradient loop                    */
    int num_solvers;    /**<  number of pardiso solvers                     */

    Solver* solverQ;
    Solver* solverQst;

    string likelihood;  /**< assumed likelihood of the observations         */
    Vect extraCoeffVecLik;

    string solver_type;

    std::string prior;  /**<  type of pripr to be used                      */

    int fct_count;      /**< count total number of function evaluations     */
    int iter_count;     /**< count total number of operator() call          */
    int iter_acc;
    Vect y;         /**<  vector of observations y. has length no.      */
    Vect theta_prior_param; /**<  vector with prior values. Constructs normal
 						      distribution with sd = 1 around these values. */
	
	Vector3i dimList;
#if 0
	Hyperparameters* theta_prior_test;
	Hyperparameters* theta_test;
	//Hyperparameters theta_prior_test = new Hyperparameters(dim_spatial_domain, manifold, dimList, theta_prior_param, theta_prior_param);
#endif
    // either Ax or B used
    SpMat Ax;           /**< sparse matrix of size no x (nu+nb). Projects 
                             observation locations onto FEM mesh and 
                             includes covariates at the end.                */
    MatrixXd B;         /**< if space (-time) model included in last 
                             columns of Ax. For regression only B exists.   */

    // used in spatial and spatial-temporal case
    SpMat c0;           /**< Diagonal mass matrix spatial part.             */
    SpMat g1;           /**< stiffness matrix space.                        */
    SpMat g2;           /**< defined as : g1 * c0^-1 * g1                   */

    // only used in spatial-temporal case
    SpMat g3;           /**< defined as : g1 * (c0^-1 * g1)^2.              */
    SpMat M0;           /**< diagonalised mass matrix time.                 */
    SpMat M1;           /**< diagonal matrix with diag(0.5, 0, ..., 0, 0.5) 
                                -> account for boundary                     */
    SpMat M2;           /**< stiffness matrix time.                         */

    SpMat Qb;           /**< setup indices once. Only prior fixed effects. */
    SpMat Qu;           /**< setup indices once. Only prior random effects */
    SpMat Qst;
    SpMat Qs;
    SpMat Qx;           /**< setup indices once. Includes Prior RE + FE.   */
    SpMat Qxy;          /**< setup indices for Qxy once. */

    double yTy;         /**< compute t(y)*y once. */
    Vect BTy;           /**< compute t(B)*y once. regression model only     */
    Vect AxTy;          /**< compute t(Ax)*y once. spat/spat temp model     */
    SpMat AxTAx;        /**< conmpute t(Ax)*Ax once. spat/spat temp model   */
    Vect mu_initial;
    Vect mu;            /**< conditional mean */
    Vect t_grad;        /**< gradient of theta */
    double min_f_theta; /**< minimum of function*/

    double w_sum;       /**< only used if validate is true                  */

    int no_f_eval;      /**< number of function evaluations per iteration   */

    MatrixXd G;         /**< orthonormal basis for finite difference stencil 
                              is Identity if smart gradient disabled        */

#ifdef SMART_GRAD
    bool thetaDiff_initialized; /**< flag in smart gradient                 */
    VectorXd theta_prev;
    MatrixXd ThetaDiff;
#endif

    const bool constr;      /**< true if there is a sum to zero constraint      */
    const MatrixXd Dx;         /**< constraint vector, sum to zero constraint      */
    const MatrixXd Dxy;         /**< constraint vector, sum to zero constraint      */

    //MatrixXd V;           /**< V = Q^-1 * t(D)                                */
    //MatrixXd W;           /**< W = A*V                                        */
    //Vect U;               /**< U = W^-1 * t(V)                                */

    const bool validate;
    const Vect w;

    bool printed_eps_flag = false;

#ifdef RECORD_TIMES
    std::string log_file_name;
    // to record times 
    double t_Ftheta_ext;
    double t_priorHyp;
    double t_priorLat;
    double t_priorLatAMat;
    double t_likel;
    double t_condLat;
    double t_condLatAMat;
    double t_thread_nom;
    double t_thread_denom;
#endif

    // these have to always be defined otherwise compile error.
    // always get measured but not written to file
    double t_priorLatChol;
    double t_condLatChol;
    double t_condLatSolve;

    double t_bfgs_iter;

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
    PostTheta(int ns, int nt, int nb, int no, 
        MatrixXd B, Vect y, 
        Vect theta_prior, 
        //Hyperparameters* theta_prior_test,
        Vect mu_initial, 
        string likelihood, Vect extraCoeffVecLik,
        string solver_type,
        const bool constr, const MatrixXd Dxy,
        const bool validate, const Vect w);
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
    PostTheta(int ns, int nt, int nb, int no, 
        SpMat Ax, Vect y, SpMat c0, SpMat g1, SpMat g2,
        Vect theta_prior, 
        Vect mu_initial, 
        string likelihood, Vect extraCoeffVecLik,
        string solver_type, 
        int dim_spatial_domain, string manifold,
        const bool constr, const MatrixXd Dx, const MatrixXd Dxy,
        const bool validate, const Vect w);

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
     * @param[in] theta_prior prior hyperparameters
     * @param[in] solver_type linear solver: currently PARDISO, BTA
     * @param[in] dim_spatial_domain dimension spatial field 1D/2D ... 
     * @param[in] manifold plane: "", "sphere", can add more later
     * @param[in] constr constraints, mainly sum-to-zero
     */ 
    PostTheta(int ns, int nt, int nb, int no, 
        SpMat Ax, Vect y, SpMat c0, SpMat g1, SpMat g2, SpMat g3, 
        SpMat M0, SpMat M1, SpMat M2, 
        Vect theta_prior, 
        Vect mu_initial, 
        string likelihood, Vect extraCoeffVecLik,
        string solver_type, 
        int dim_spatial_domain, string manifold,
        const bool constr, const MatrixXd Dx, const MatrixXd Dxy,
        const bool validate, const Vect w); 

    /**
     * @brief constructor for spatial temporal model w/ add. spatial field
     * @brief constructor for both spatial models (order 2).
     * @param[in] ns_ number of spatial grid points per time step.
     * @param[in] nt_ number of temporal time steps.
     * @param[in] nss_ number of spatial grid points in add. spatial field
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
    PostTheta(int ns, int nt, int nss, int nb, int no, 
        SpMat Ax, Vect y, SpMat c0, SpMat g1, SpMat g2, SpMat g3, 
        SpMat M0, SpMat M1, SpMat M2, 
        Vect theta_prior_param, 
        Vect mu_initial, 
        string likelihood, Vect extraCoeffVecLik,
        string solver_type, 
        int dim_spatial_domain, string manifold,
        const bool constr, const MatrixXd Dx, const MatrixXd Dxy, 
        const bool validate, const Vect w);


    /**
     * @brief structure required by BFGS solver, requires : theta, gradient theta
     * \note Gradient call is already parallelised using nested OpenMP. 
     * --> there are l1 threads (usually 8, one for each function evaluation), that themselves
     * then split into another e.g. 8 threads, when calling PARDISO to factorise the system.
     * --> somehow introduce additional parallelism to compute f(theta), possible to do in parallel
     */
    double operator()(Vect& theta, Vect& grad);

#ifdef DATA_SYNTHETIC
    double compute_error_bfgs(Vect& theta);
#endif

	//Hyperparameters create_hp(Vect param, char scale);

    /**
     * @brief overwriting G every time, not explicitly listed, better way to do this? needs to be 
     * stored after every iteration for smart hessian ...       */
    void computeG(Vect& theta);

    int get_fct_count();

    // ============================================================================================ //
    // CONVERT MODEL PARAMETRISATION TO INTERPRETABLE PARAMETRISATION & VICE VERSA

	void convert_theta2interpret(Vect& theta, Vect& theta_interpret);

	void convert_interpret2theta(Vect& theta_interpret, Vect& theta);
	
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
	void convert_theta2interpret_spatTemp(double lgamE, double lgamS, double lgamT, double& sigU, double& ranS, double& ranT);
	
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
	void convert_interpret2theta_spatTemp(double sigU, double ranS, double ranT, double& lgamE, double& lgamS, double& lgamT);

    /**
     * @brief convert hyperparameters theta from the interpretable parametrisation to the
     * model parametrisation ie. from log(rangeS, sigma.u) to log(gamma_s, gamma_E) for spatial model order 2
     * @param [in]      log(ranS)    spatial range
     * @param [in]      log(sigma.u) precision of random effects
     * @param [inout]   log(gamma_s)
     * @param [inout]   log(gamma_E) 
     */
    void convert_interpret2theta_spat(double lranS, double lsigU, double& lgamS, double& lgamE);

    /**
     * @brief convert hyperparameters theta from the interpretable parametrisation to the
     * model parametrisation ie. from log(rangeS, sigma.u) to log(gamma_s, gamma_E) for spatial model order 2
     * @param [in]          log(gamma_s)
     * @param [in]          log(gamma_E) 
     * @param [inout]       log(ranS)    spatial range
     * @param [inout]       log(sigma.u) precision of random effects
     */
    void convert_theta2interpret_spat(double lgamS, double lgamE, double& lranS, double& lsigU);


    // ============================================================================================ //
    // FUNCTIONS TO BE CALLED AFTER THE BFGS SOLVER CONVERGED

    /**
     * @brief get conditional mean mu for theta -- Gaussian case.
     * @param [in]    theta hyperparameter vector
     * @param [inout] mu vector of the conditional mean
     */ 
    void get_mu(Vect& theta, Vect& mu);

    #if 0
    /**
     * @brief get conditional mean mu for non-Gaussian case with no hyperparamters.
     * @param [in]    extraCoeffVecLik extra coefficients likelihood.
     * @param [inout] mu vector of the conditional mean
     */ 
    void get_mu(Vect& mu);
    #endif

    /**
     * @brief returns current gradient of theta.
     * @return gradient_theta
     */ 
    Vect get_grad();

    void get_Qprior(Vect theta, SpMat& Qprior);

    /**
     * @brief Compute Covariance matrix of hyperparameters theta, at theta.
     * @details computes the hessian of f(theta) using a second order finite
     * difference stencil and then inverts the hessian. Gaussian assumption.
     * @param [in] theta hyperparameter Vector
     * @return cov covariance matrix of the hyperparameters
     */ 
    MatrixXd get_Covariance(Vect theta, double eps);

    MatrixXd get_Cov_interpret_param(Vect interpret_theta, double eps);

    double f_eval(Vect& theta);

    /**
     * @brief Compute the marginal variances of the latent parameters at theta. 
     * Using selected inversion procedure.
     * @param[in]    Vector theta.
     * @param[inout] Vector with marginals of f.
     */ 
    void get_marginals_f(Vect& theta, Vect& mu, Vect& vars);

    /**
     * @brief Compute the marginal variances of the latent parameters at theta. 
     * Using selected inversion procedure.
     * @param[in]    Vector theta.
     * @param[inout] Vector with selected inverse for all non-zero entries of Q.
     */ 
    void get_fullFact_marginals_f(Vect& theta, SpMat& Qinv);

    void compute_fullInverseQ(Vect& theta, MatrixXd& Qinv);

    /**
     * @brief computes the hessian at theta using second order finite difference.
     * Is used be get_Covariance.
     * @param[in] theta hyperparameter vector
     * @return Dense Matrix with Hessian. 
     * \todo not yet parallelised .... 
     */ 
    MatrixXd hess_eval(Vect theta, double eps);

    MatrixXd hess_eval_interpret_theta(Vect interpret_theta, double eps);

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
     * @return       f(theta) value
     */
    double eval_post_theta(Vect& theta, Vect& mu);

    /**
     * @brief evaluate log prior of the hyperparameters using original theta value
     * @param[in] thetai current theta_i value
     * @param[in] thetai_original original theta_i value
 	 * @param[inout] log prior is being updated.
 	 * @details variance / precision of 1 : no normalising constant. 
 	 * computed through -0.5 * (theta_i* - theta_i)*(theta_i*-theta_i) 
     */	
	void eval_log_gaussian_prior_hp(Vect& theta_param, Vect& theta_prior_param, double& log_prior);

    /**
     * @brief evaluate log prior using PC prior 
     * @param[inout] log sum     
     * @param[in] lambda : parameters for penalised complexity prior
     * @param[in] theta_interpret current theta value in interpretable scale
     * @details complicated prior. check appropriate references for details.
     */ 
    void eval_log_pc_prior_hp(double& log_sum, Vect& lambda, Vect& theta_interpret);

    void update_mean_constr(const MatrixXd& D, Vect& e, Vect& sol, MatrixXd& V, MatrixXd& W, MatrixXd& U, Vect& updated_sol);

    void eval_log_dens_constr(Vect& x, Vect& mu, SpMat&Q, double& log_det_Q, const MatrixXd& D, MatrixXd& W, double& val_log_dens);

    /**
     * @brief evaluate log prior of random effects
     * @param[in] theta current theta vector
         * @param[inout] log_det inserts log determinant.
         * \todo construct spatial matrix (at the moment this is happening twice. FIX)
     */ 
    void eval_log_prior_lat(Vect& theta, double &val);

    /**
     * @brief compute log likelihood : log_det tau*no and value -theta*yTy
     * @param[in] theta current theta vector
     * @param[inout] log_det inserts log determinant of log likelihood.
     * @param[inout] val inserts the value of -theta*yTy
     */ 
    void eval_likelihood(Vect& theta, double &log_det, double &val);
    
    /**
     * @brief spatial model : SPDE discretisation -- matrix construction
     * @param[in] theta current theta vector
     * @param[inout] Qs fills spatial precision matrix
     */
    void construct_Q_spatial(Vect& theta, SpMat& Qs);

    /**
     * @brief spatial temporal model : SPDE discretisation. DEMF(1,2,1) model.
     * @param[in] theta current theta vector
     * @param[inout] Qst fills spatial-temporal precision matrix
     */
    void construct_Q_spat_temp(Vect& theta, SpMat& Qst);

    void construct_Qprior(Vect& theta, SpMat& Qx);

    /** @brief construct precision matrix. 
     * Calls spatial, spatial-temporal, etc.
     * @param[in] theta current theta vector
     * @param[in] mu     mode latent parameters
     * @param[inout] Q fills precision matrix
     */
    void construct_Q(Vect& theta, Vect& mu, SpMat& Q);

    /** @brief Assemble right-handside. 
     * @param[in] theta current theta vector
     * @param[inout] rhs right-handside
     * /todo Could compute Ax^T*y once, and only multiply with appropriate exp_theta.
     */ 
    void construct_b(Vect& theta, Vect &rhs);

    //void update_mean_constr(MatrixXd& D, Vect& e, Vect& sol, MatrixXd& V, MatrixXd& W);

    /** @brief Evaluate denominator: conditional probability of Qx|y
     * @param[in] theta current theta vector
     * @param[inout] log_det fill log determinant of conditional distribution of denominator
     * @param[inout] val fill value with mu*Q*mu
     * @param[inout] Q construct precision matrix
     * @param[inout] rhs construct right-handside
     * @param[inout] mu insert mean of latent parameters
     */
    void eval_denominator(Vect& theta, double& val, SpMat& Q, Vect& rhs, Vect& mu);

    // ============================================================================================ //
    // INNER ITERATION & everything that is needed for it

    /** @brief evaluate Gaussian log prior (without log determinant!!), mean assumed to be zero
     * @param[in] Qprior precision matrix
     * @param[in] x current x vector
     * @param[out] f_val evaluated log density
     */
    double cond_LogPriorLat(SpMat& Qprior, Vect& x);

    /** @brief evaluate log Poisson likelihood
     * @param[in] eta Vector. linear predictor eta = A*x
     * @param[out] f_val double. evaluated log density
     */
    double cond_LogPoisLik(Vect& eta);

    /** @brief evaluate negative log Poisson likelihood
     * @param[in] eta Vector. linear predictor eta = A*x
     * @param[out] f_val double. evaluated negative log density
     */
    double cond_negLogPoisLik(Vect& eta);

    /** @brief evaluate analytical negative gradient log Poisson likelihood
     * @param[in] eta Vector. linear predictor eta = A*x
     * @param[out] grad Vect. gradient.
     */
    Vect grad_cond_negLogPoisLik(Vect& eta);

    /** @brief evaluate analytical negative diagonal Hessian of log Poisson likelihood
     * @param[in] eta Vector. linear predictor eta = A*x
     * @param[out] diagHess Vect. diagonal of Hessian (off-diagonal entries are zero)
     */
    Vect diagHess_cond_negLogPoisLik(Vect& eta);


    /** @brief evaluate negative condiational log Poisson + Gaussian prior
     * @param[in] Qprior SpMat. precision matrix.
     * @param[in] x Vector. current vector x.
     * @param[out] f_val double. evaluated negative log density
     */
    double cond_negLogPois(SpMat& Qprior, Vect& x);

    /** @brief link function. vectorized evaluation of sigmoid function for each entry
     * @param[in] x Vector. current vector x.
     * @param[inout] sigmoidX Vector. sigmoid(x) element-wise.
     */
    void link_f_sigmoid(Vect& x, Vect& sigmoidX);

    /** @brief evaluate negative log Binomial likelihood
     * @param[in] extraCoeffVecLik Vector. ntrials.
     * @param[in]  eta Vector. linear predictor eta = A*x.
     * @param[out] f_val double. evaluated negative log density.
     */
    double cond_negLogBinomLik(Vect& eta);

    /** @brief evaluate negative condiational log Poisson + Gaussian prior
     * @param[in] extraCoeffVecLik Vector. ntrials.
     * @param[in] Qprior SpMat. precision matrix.
     * @param[in] x Vector. current vector x.
     * @param[out] f_val double. evaluated negative log density
     */
    double cond_negLogBinom(SpMat& Qprior, Vect& x);

    /** @brief evaluate negative condiational log likelihood + Gaussian prior
     * @param[in] extraCoeffVecLik Vector. 
     * @param[in] Qprior SpMat. precision matrix.
     * @param[in] x Vector. current vector x.
     * @param[in] lik_func function. defines the likelihood
     * @param[out] f_val double. evaluated negative log density
     */
    double cond_negLogDist(SpMat &Qprior, Vect& x, function<double(Vect&, Vect&)> lik_func);

    /** @brief compute finite difference gradient. 1st order central difference. currently stepsize h fixed.
     * @param[in] extraCoeffVecLik Vector. 
     * @param[in] eta Vector. linear predictor eta = A*x.
     * @param[in] lik_func function. defines the likelihood
     * @param[inout] grad Vector. gradient.
    */
    void FD_gradient(Vect& eta, Vect& grad);

    /** @brief compute finite difference diagonal of hessian. 2nd order central difference. currently stepsize h fixed.
     * @param[in] extraCoeffVecLik Vector. 
     * @param[in] eta Vector. linear predictor eta = A*x.
     * @param[in] lik_func function. defines the likelihood
     * @param[inout] diag_hess Vector. diagonal of Hessian.
    */
    void FD_diag_hessian(Vect& eta, Vect& diag_hess);

    /**
     * @brief Newton iteration to find optimum of conditional distribution latent parameters of prior & likelihood
     * @param[in] theta hyperparameters. can be an empty vector. 
     * @param[inout] x latent parameters. contains initial guess of mode on entry and found mode on exit.
     * @param[inout] Q SpMat. precision matrix.
     * @param[inout] x log det of Q.
     */
    void NewtonIter(Vect& theta, Vect& x, SpMat& Q, double& log_det);


    // measure times within each iterationstd::string file_name, int& iter_count, double& t_Ftheta_ext, double& t_priorHyp, 
    void record_times(std::string file_name, int iter_count, double t_Ftheta_ext, double t_thread_nom, double t_priorHyp, 
                                double t_priorLat, double t_priorLatAMat, double t_priorLatChol, double t_likel, 
                                double t_thread_denom, double t_condLat, double t_condLatAMat, double t_condLatChol, double t_condLatSolve);


     /**
     * @brief class destructor. Frees memory allocated by PostTheta class.
     */
    ~PostTheta();

};

#endif


