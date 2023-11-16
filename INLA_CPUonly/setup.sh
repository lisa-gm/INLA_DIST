# setup script that is source by .bashrc to set libraries paths etc.

# MKL
source /opt/intel/oneapi/mkl/latest/env/vars.sh intel64
#source /opt/intel/mkl/bin/mklvars.sh intel64

# MAGMA
export LD_LIBRARY_PATH=/home/x_gaedkelb/applications/magma-2.5.4/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/home/x_gaedkelb/applications/magma-2.5.4/lib:$LIBRARY_PATH
export CPATH=/home/x_gaedkelb/applications/magma-2.5.4/include:$CPATH

# PARDISO
export LD_LIBRARY_PATH=/home/x_gaedkelb/applications/pardiso:$LD_LIBRARY_PATH
export LIBRARY_PATH=/home/x_gaedkelb/applications/pardiso:$LIBRARY_PATH
