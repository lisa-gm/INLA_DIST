
example: example.cpp
	g++ $< -o $@ -I/usr/include/eigen3 -lm -I/home/x_gaedkelb/applications/LBFGSpp/include -I/usr/include/suitesparse -lcholmod -lm -std=c++17
	
#g++ -I/path/to/eigen -I/path/to/lbfgspp/include -O2 example.cpp

test_file: test_file.cpp
	g++ $< -o $@ -I/usr/include/eigen3 -lm -I/home/x_gaedkelb/applications/LBFGSpp/include -I/usr/include/suitesparse -lcholmod -lm -std=c++17
	


clean: 
	rm -f example
	rm -f test_file
