#ifndef __UTILITIES
#define __UTILITIES

#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <ctime>
#include <math.h>
#include <cmath>
#include "Blas.H"

void icopy(int,int*,int*);

int Round(double);

double get_time(double);

/************************************************************************************************/

template <typename T>
void init_var(T *var,int N);

template <typename T>
void init_var(T *var,int N)
{
    for(int i=0;i<N;i++){
        var[i] = (T)0;
    }
}

/************************************************************************************************/

template <typename T,typename W>
T convert(W val);

template<>
inline double convert(CPX val)
{
    return real(val);
}

template<>
inline CPX convert(CPX val)
{
    return val;
}

template<>
inline double convert(double val)
{
    return val;
}

template<>
inline CPX convert(double val)
{
    return CPX(val,0.0);
}

/************************************************************************************************/

#endif

