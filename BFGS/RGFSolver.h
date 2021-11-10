#ifndef RGFSOLVER_H
#define RGFSOLVER_H

#include "Solver.h"

#if 0
typedef CPX T;
#define assign_T(val) CPX(val, 0.0)
#else
typedef double T;
#define assign_T(val) val
#endif

//extern "C" double 

 /**
 * @brief creates solver class using RGF-GPU for factorising, solving and selectively inverting linear system.
 * @details divided into set up, symbolic factorisation, numerical factorisation, numerical factorisation & solve 
 * and selected inversion (of the diagonal elements)
 */
class RGFSolver: public Solver {

    private:

        /* matrix size */
        unsigned int n;                  /**< size of the matrix */
        unsigned int nnz;       /**< number of nonzeros */

        // to avoid redeclaration every time
        size_t i;

        size_t ns;
        size_t nt;
        size_t nb;
        size_t no;

        SpMat Q;                /**< sparse precision matrix Q. Eigen format. */

        int* ia;                /**< CSR format. row indices. */
        int* ja;                /**< CSR format. col pointers. */
        double* a;              /**< CSR format. values. */

        double* b;              /**< right-hand side. */
        double* x;              /**< placeholder for solution. */
   	public:
   		RGFSolver(size_t ns_, size_t nt_, size_t nb_, size_t no_);

		void symbolic_factorization(SpMat& Q, int& init);

		void factorize(SpMat& Q, double& log_det);

		void factorize_solve(SpMat& Q, Vector& rhs, Vector& sol, double &log_det);

      	void selected_inversion(SpMat& Q, Vector& inv_diag);

      	// will also need a "simple inversion" method to independent of PARDISO. regular lapack should do (see pardiso)

      	~RGFSolver();

};


#endif