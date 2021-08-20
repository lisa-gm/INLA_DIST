#ifndef RGFSOLVER_H
#define RGFSOLVER_H

#include "Solver.h"

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