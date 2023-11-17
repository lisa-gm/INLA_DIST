#include "Utilities.H"

void icopy(int N, int* x, int* x_copy)
{
    int i;

    for(i=0;i<N;i++){
        x_copy[i]=x[i];
    }
    
}

/************************************************************************************************/

int Round(double x)
{
    if(x<0){
        return (int)(x-0.5);
    }else{
        return (int)(x+0.5);
    }
}

/************************************************************************************************/

double get_time(double t0)
{
    timeval tim;
    gettimeofday(&tim, NULL);
    
    return (double)(tim.tv_sec+(tim.tv_usec/1000000.0))-t0;
}

/************************************************************************************************/
