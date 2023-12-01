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

The code itself is documented using doxygen [here](https://lisa-gm.github.io/INLA_DIST/documentation/html/index.html).
- PostTheta       : contains the main parts of the INLA methodology.
- BTASolver       : contains the interface to the BTA solver.
- PardisoSolver   : contains the interface to PARDISO.
- EigenSolver     : contains the interface to Eigen's internal solver (only suitable for very small test cases).

During runtime you can choose between BTA solver (check for installation details), PARDISO & Eigen's internal solver (with and without CHOLMOD support). For examples see one of the different runscripts. 


### Simple Example

The executable expects inputs of the following form: 
```
mpirun -n ${num_ranks} ./call_INLA ${ns} ${nt} ${nss} ${nb} ${no} ${likelihood} ${folder_path} ${solver_type}
```
In the ```folder_path``` folder must contain the generated matrices of the form as described in Rscripts.

The small dataset that is included can be called as follows ``` mpirun -np 1 ./call_INLA 42 3 0 2 126 gaussian ../data/synthetic/ns42_nt3_nb2 Eigen``` using Eigen's internal solver.

The output should then look something like this: 

```
============== PARALLELISM & NUMERICAL SOLVERS ==============
total no MPI ranks  : 1
OMP threads level 1 : 2
OMP threads level 2 FIXED TO 1!!
available GPUs      : 1
Assumed likelihood of the data: gaussian
Solver : Eigen

==================== MODEL SPECIFICATIONS ===================
spatial-temporal model. dim(theta) = 4
read in all matrices.
sum(y) = -39.5253
using SYNTHETIC DATASET
spatial domain: sphere

call spatial-temporal constructor.
likelihood: gaussian
Eigen Solver. NO CHOLMOD.
Eigen Solver. NO CHOLMOD.
Prior : pc
Regular FD gradient enabled.

======================= HYPERPARAMETERS =====================
theta prior param       : 0.233333    0.098      0.7 0.233333
theta interpret. param. : 1.366087 2.350673 0.030923 1.405511
initial theta           :  1.366087  1.248857 -1.310952 -3.284129

====================== CALL BFGS SOLVER =====================
theta param : 1.366087 2.350673 0.030923 1.405511
theta       :  1.366087  1.248857 -1.310952 -3.284129
lineserach condition : Wolfe. m = 6, delta = 0.001000, epsilon = 0.100000, rel eps = 0.001000, max iter = 200
Finite Difference    : h = 1.000000e-03
theta interpret: 1.3661 2.3507 0.0309 1.4055    f_theta: 4972.9190
time f + grad f eval : 0.0068
theta interpret: 0.3743 2.3633 0.0555 1.5579    f_theta: 1929.8837
time f + grad f eval : 0.0058
theta interpret: -0.2519  2.3630  0.0551  1.5797    f_theta: 1085.4455
time f + grad f eval : 0.0057
theta interpret: -0.9758  2.3538  0.0378  1.5424    f_theta: 595.2525

...

...

theta interpret: 1.4452 0.1847 2.2992 1.3342    f_theta: 223.0946
time f + grad f eval : 0.0056
theta interpret: 1.4489 0.1818 2.2906 1.3322    f_theta: 223.0941
time f + grad f eval : 0.0056
exited. epsilon = 0.1000, epsilon_rel * xnorm = 0.0066
gnorm = 0.0632
44 iterations and 891 fn calls.

f(x)                         : 223.0941
grad                         : -0.0573 -0.0158 -0.0004 -0.0214

orig. mean parameters        :  1.3863 -5.8825  1.0397  3.6889
est.  mean parameters        :  1.4489 -5.4285  0.8579  3.3133

orig. mean interpret. param. :  1.3863 -0.0000  2.3026  1.4063
est.  mean interpret. param. :  1.4489 0.1818 2.2906 1.3322
covariance                   :
 0.9408 -0.0222  0.0209 -0.0083
-0.0222  0.0801 -0.0581 -0.0610
 0.0209 -0.0581  0.0590  0.0316
-0.0083 -0.0610  0.0316  0.0656
time get covariance          : 0.0204 sec
estimated hessian         :
  1.1306   0.3718   0.8384  -3.4283
  0.3718  68.8108 -25.1295 -42.5444
  0.8384 -25.1295  23.9811 -27.7527
 -3.4283 -42.5444 -27.7527 234.1732
eigenvalues hessian :
246.5660
  1.0584
 76.0359
  4.4354

covariance interpr. param.  :
 0.9408 -0.0209 -0.0501  0.0040
-0.0209  0.0590  0.0864  0.0207
-0.0501  0.0864  0.1754  0.0358
 0.0040  0.0207  0.0358  0.0123

sd hyperparameters interpr. :   0.9699 0.2429 0.4188 0.1110

estimated mean fixed effects : -0.3184  3.4312
estimated mean random effects: -0.9788  5.6212 -0.1274  2.5663  0.7771 -6.5521 -3.5435 -3.4813  4.9832  4.3741

==================== compute marginal variances ================

USING ESTIMATED THETA :  1.4489 -5.4285  0.8579  3.3133
est. standard dev fixed eff  : 1.8574 0.2910
est. std dev random eff      : 1.9345 1.9185 1.9182 1.9457 1.9175 1.9167 1.9170 1.9421 1.9452 1.9498

total number fn calls        : 958

time BFGS solver             : 0.5614 sec
time get covariance          : 0.0207 sec
time get marginals FE        : 0.0016 sec
total time                   : 1.4809 sec
```


