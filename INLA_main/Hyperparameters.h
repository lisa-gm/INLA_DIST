#ifndef __HYPERPARAMETER_H
#define __HYPERPARAMETER_H

#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include "mpi.h"

#include <Eigen/Core>
#include <Eigen/Dense>

using Eigen::MatrixXd;
typedef Eigen::VectorXd Vect;

using namespace Eigen;


// make hyperparameters a class with different attributes & conversion functions for hyperparameters
// TODO: make such that when theta is update in one scale the other gets udpated as well??
class Hyperparameters {
    public:

    int spatial_dim      = 2;
    std::string manifold = "";
    
    Vect lik; // log
    Vector2d spatF_modelS         = Vector2d::Zero();  // log hp related to spatial field in model scale. order:    
    Vector2d spatF_interpretS     = Vector2d::Zero();  // log hp related to spatial field in model scale. order: range s, sigma s

    Vector3d spatTempF_modelS     = Vector3d::Zero();  // log hp related to spatial temporal field in model scale. order: 
    Vector3d spatTempF_interpretS = Vector3d::Zero();  // log hp related to spatial temporal field in model scale. order: range s, range t, sigma st
    
    // order lik, spatTempF, spatF
    Vector3i dimList              = Vector3i::Zero();  // contains dim of each of the above: initialized to zero. update when above are being set.

    char scale; 
    Vect& flat;

    //Vect& flat_modelS;
    //Vect& flat_interpretS;

    // ============================================================== //

    Hyperparameters(int spatial_dim_, std::string manifold_, Vector3i dimList_, char scale_, Vect& flat_);
    //Hyperparameters(int spatial_dim_, std::string manifold_, Vector3i dimList_, Vect& flat_modelS_, Vect& flat_interpretS_);
    //Hyperparameters(int spatial_dim_, std::string manifold_, Vector3i dimList_);

    ~Hyperparameters();

    void convert_theta2interpret();
    void convert_interpret2theta();

    Vect flatten_modelS();     // glue all existing hyperparameter elements together in model scale
    Vect flatten_interpretS(); // glue all existing hyperparameter elements together in interpret scale

    void update_modelS(Vect theta_update);
    void update_interpretS(Vect theta_update);
};


#endif // endif IFNDEF HYPERPARAMETER