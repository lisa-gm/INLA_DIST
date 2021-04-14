# makefile
export LD_LIBRARY_PATH=/home/x_gaedkelb/applications:$LD_LIBRARY_PATH

regression_model: regression_model.cpp
	g++ $< -o $@ -I/usr/include/eigen3 -I/usr/include/suitesparse -lcholmod -lm

read_write_functions: read_write_functions.cpp
	g++ $< -o $@ -I/usr/include/eigen3 -lm



