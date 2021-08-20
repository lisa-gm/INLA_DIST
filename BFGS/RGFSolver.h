#ifndef RGFSOLVER_H
#define RGFSOLVER_H

#include "Solver.h"


 /**
 * @brief creates solver class using RGF-GPU for factorising, solving and selectively inverting linear system.
 * @details divided into set up, symbolic factorisation, numerical factorisation, numerical factorisation & solve 
 * and selected inversion (of the diagonal elements)
 */
class RGFSolver: public Solver {
   	public:
   		RGFSolver();

		void symbolic_factorization(SpMat& Q, int& init);

		void factorize(SpMat& Q, double& log_det);

		void factorize_solve(SpMat& Q, Vector& rhs, Vector& sol, double &log_det);

      	void selected_inversion(SpMat& Q, Vector& inv_diag);

      	//~RGFSolver();

    private: 
    	int ns;

};


#endif