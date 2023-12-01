## INLA<sub>main</sub>

### Compiling 

The following packages or software libraries are necessary for compilation. 

- OpenMP
- MPI
- [Intel oneAPI Math Kernel Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html#gs.0rz4o1)
- [Eigen](https://eigen.tuxfamily.org)
- [adapted LBFGSpp](https://github.com/lisa-gm/adapted_LBFGSpp)

The BTA solver requires CUDA + MAGMA, whereas the sparse CPU solver requires PARDISO, see main README, BTA and $\text{INLA}_{\text{CPUonly}}$ subfolder for more details. Once this are installed you it is necessary to update the respective paths in the makefile. ```BASE_PATH``` should point to your home/main directory. You will also need to add the paths to the above packages.  Note, that you might also have to add them to your ```$CPATH, $LIBRARY_PATH``` and ```$LD_LIBRARY_PATH```. Once this is done you can compile by typing

```make```

in this folder.

### Running 

```main: call_INLA.cpp``` 

There are different run scripts for different machines available which you can adapt. Different input matrices are expected in the ```../data``` folder. A small test example can be found there already. It was generated using ```generate_synth_dataset.R``` from the Rscripts folder.

The code itself is documented using doxygen. 
- PostTheta       : contains the main parts of the INLA methodology.
- BTASolver       : contains the interface to the BTA solver.
- PardisoSolver   : contains the interface to PARDISO.
- EigenSolver     : contains the interface to Eigen's internal solver (only suitable for very small test cases).

During runtime you can choose between BTA solver (check for installation details), PARDISO & Eigen's internal solver. For examples see one of the different runscripts. 


### Simple Example

``` 

```

The output should look something like this 


