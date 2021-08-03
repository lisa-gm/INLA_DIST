

pardiso_reuse_symf_ex1: pardiso_reuse_symf_ex1.cpp
	g++ $< -o $@ -fopenmp /home/x_gaedkelb/applications/pardiso/libpardiso700-GNU840-X86-64-RINLA.so \
	-L${MKLROOT}/lib/intel64 -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -lgomp -lgfortran -fPIC  -fopenmp -fPIC -lm


pardiso_reuse_symf_ex2: pardiso_reuse_symf_ex2.cpp
	g++ $< -o $@ -I/usr/include/eigen3 -fopenmp /home/x_gaedkelb/applications/pardiso/libpardiso700-GNU840-X86-64-RINLA.so \
	-L${MKLROOT}/lib/intel64 -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -lgomp -lgfortran -fPIC  -fopenmp -fPIC -lm



make clean:
	rm -f pardiso_reuse_symf_ex1
	rm -f pardiso_reuse_symf_ex2