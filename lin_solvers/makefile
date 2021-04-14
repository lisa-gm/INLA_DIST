# makefile

eigen_lin_solver: eigen_lin_solver.cpp
	g++ -I eigen_lin_solver.cpp -o eigen_lin_solver

cholmod_eigen: cholmod_eigen.cpp
	g++ $< -I/usr/include/eigen3 -I/usr/include/suitesparse -o $@ -lcholmod -lm
