#ifndef RGFSOLVER_H
#define RGFSOLVER_H

#include "Solver.h"


class RGFSolver: public Solver {
   	public:
		RGFSolver(int ns){
		 std::cout << "constructing RGF solver. ns = " << ns << std::endl;
		}

		void symbolic_factorization(SpMat& Q, int& init) {
			init = 1;
			std::cout << "Placeholder SYMBOLIC_FACTORIZATION()." << std::endl;
		}

		void factorize(SpMat& Q, double& log_det) {
			log_det = 1;
			std::cout << "Placeholder FACTORIZE()." << std::endl;

		}

		void factorize_solve(SpMat& Q, Vector& rhs, Vector& sol, double &log_det) {
			sol.setOnes();
			std::cout << "Placeholder FACTORIZE_SOLVE()." << std::endl;
		}

      void selected_inversion(SpMat& Q, Vector& inv_diag) {
        	inv_diag = 5*Vector::Ones(inv_diag.size());
        	std::cout << "Placeholder SELECTED_INVERSION()." << std::endl;

      }

    private: 
    	int ns;

};



#endif