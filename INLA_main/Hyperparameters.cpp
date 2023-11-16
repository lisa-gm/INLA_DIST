#include "Hyperparameters.h"

Hyperparameters::Hyperparameters(int spatial_dim_, std::string manifold_, Vector3i dimList_, char scale_, Vect& flat_): spatial_dim(spatial_dim_), manifold(manifold_), dimList(dimList_), scale(scale_), flat(flat_){
//Hyperparameters::Hyperparameters(int spatial_dim_, std::string manifold_, Vector3i dimList_, Vect& flat_modelS_, Vect& flat_interpretS_): spatial_dim(spatial_dim_), manifold(manifold_), dimList(dimList_), flat_modelS(flat_modelS_), flat_interpretS(flat_interpretS_){
        
        if(scale != 'm' || scale != 'i'){
            printf("invalid scale! choices are i or m!\n");
            exit(1);
        }
        lik.resize(dimList(0));
        flat.resize(dimList.sum());

        //flat_modelS.resize(dimList.sum());
        //flat_interpretS.resize(dimList.sum());
} 

Hyperparameters::~Hyperparameters(){
         printf("dummy destructor.\n");
    };

void Hyperparameters::update_modelS(Vect theta_update){
    // check dimList -> assumes matching input
    //std::cout << "in update modelS. dimList: " << dimList.transpose() << std::endl;
    //std::cout << "theta update: " << theta_update.transpose() << std::endl;
    
    int count = 0;
    for(int i=0; i<dimList(0); i++){
        lik(i) = theta_update(count);
        count += 1;
    }

    for(int i=0; i<dimList(1); i++){
        spatTempF_modelS(i) = theta_update(count);
        count += 1;
    }

    for(int i=0; i<dimList(2); i++){
        spatF_modelS(i) = theta_update(count);
        count += 1;
    }

    convert_theta2interpret();

    if(scale == 'm'){
        flat = flatten_modelS();
    }else if(scale == 'i'){
        flat = flatten_interpretS();
    } else {
        printf("invalid scale parameter!\n");
        exit(1);
    }

    //flat_interpretS = flatten_interpretS();
    //flat_modelS     = flatten_modelS();
}

void Hyperparameters::update_interpretS(Vect theta_update){
    // check dimList -> assumes matching input
    
    int count = 0;
    for(int i=0; i<dimList(0); i++){
        lik(i) = theta_update(count);
        count += 1;
    }

    for(int i=0; i<dimList(1); i++){
        spatTempF_interpretS(i) = theta_update(count);
        count += 1;
    }

    for(int i=0; i<dimList(2); i++){
        spatF_interpretS(i) = theta_update(count);
        count += 1;
    }

    convert_interpret2theta();

    if(scale == 'm'){
        flat = flatten_modelS();
    }else if(scale == 'i'){
        flat = flatten_interpretS();
    } else {
        printf("invalid scale parameter!\n");
        exit(1);
    }

    //flat_interpretS = flatten_interpretS();
    //flat_modelS     = flatten_modelS();
}

// flatten. order: lik, spatTemp, spat
Vect Hyperparameters::flatten_modelS(){
    Vect theta_list(dimList.sum());
    theta_list.setZero();

    int count = 0;
    
    for(int i=0; i<dimList(0); i++){
        theta_list(count) = lik(i);
        count += 1;
    }
    
    for(int i=0; i<dimList(1); i++){
        theta_list(count) = spatTempF_modelS(i);
        count += 1;
    }

    for(int i=0; i<dimList(2); i++){
        theta_list(count) = spatF_modelS(i);
        count += 1;
    }

    //std::cout << "theta_list : " << theta_list.transpose() << std::endl;
    return theta_list;
}

// flatten. order: lik, spatTemp, spat
Vect Hyperparameters::flatten_interpretS(){
    Vect theta_list(dimList.sum());

    int count = 0;
    
    for(int i=0; i<dimList(0); i++){
        theta_list(count) = lik(i);
        count += 1;
    }
    
    for(int i=0; i<dimList(1); i++){
        theta_list(count) = spatTempF_interpretS(i);
        count += 1;
    }

    for(int i=0; i<dimList(2); i++){
        theta_list(count) = spatF_interpretS(i);
        count += 1;
    }

    //std::cout << "theta interpret scale : " << theta_list.transpose() << std::endl;
    return theta_list;
}

// take variables from spatTemp_modelS: convert & overwrite spatTem_interpretS
void Hyperparameters::convert_theta2interpret(){ 
    //printf("in convert theta2interpret()\n");

    // spatial temporal parameters. assumed order:     
    //double lgamE, double lgamS, double lgamT, double& ranS, double& ranT, double& sigU
    if(dimList(1) > 0){
        double alpha_t = 1; 
        double alpha_s = 2;
        double alpha_e = 1;

        double alpha = alpha_e + alpha_s*(alpha_t - 0.5);
        //double nu_s  = alpha   - 1;
        double nu_s  = alpha - 1; 
        double nu_t  = alpha_t - 0.5;

        double gE = exp(spatTempF_modelS(0)); // lgamE
        double gS = exp(spatTempF_modelS(1)); // lgamS
        double gT = exp(spatTempF_modelS(2)); // lgamT

        double ranS = log(sqrt(8*nu_s)/gS);
        double ranT = log(gT*sqrt(8*nu_t)/(pow(gS, alpha_s)));

        double sigU;
        
        if(manifold == "sphere"){
            double cR_t = std::tgamma(alpha_t - 1.0/2.0)/(std::tgamma(alpha_t)*pow(4*M_PI, 1.0/2.0));
            double cS = 0.0;
            for(int k=0; k<50; k++) // compute 1st 100 terms of infinite sum
            {  
                cS += (2.0*k + 1) / (4*M_PI* pow(pow(gS, 2) + k*(k+1), alpha));
            }
            //printf("cS : %f\n", cS);
            sigU = log(sqrt(cR_t*cS)/(gE*sqrt(gT)));

        } else {
            double c1_scaling_const = pow(4*M_PI, spatial_dim/2.0) * pow(4*M_PI, 1.0/2.0); // second for temporal dim
            //double c1_scaling_const = pow(4*M_PI, 1.5);
            //std::cout << "c1_scaling_const theta2interpret : " << c1_scaling_const << std::endl;	
            double c1 = std::tgamma(nu_t)*std::tgamma(nu_s)/(std::tgamma(alpha_t)*std::tgamma(alpha)*c1_scaling_const);
            sigU = log(sqrt(c1)/((gE*sqrt(gT))*pow(gS,alpha-spatial_dim/2)));
        }

        spatTempF_interpretS(0) = ranS;
        spatTempF_interpretS(1) = ranT;
        spatTempF_interpretS(2) = sigU;

    }

    // spatial parameters. assumed order:
    // double lgamE, double lgamS, double& lranS, double& lsigU
    if(dimList(2) > 0){
        double alpha    = 2.0;
        double nu_s     = alpha - spatial_dim/2.0;

        double lgamE = spatF_modelS(0);
        double lgamS = spatF_modelS(1);

        double lsigU = 0.5*(std::tgamma(nu_s) - (std::tgamma(alpha) + 0.5*spatial_dim*log(4*M_PI) + 2*nu_s*lgamS + 2*lgamE));
        double lranS = 0.5*log(8*nu_s) - lgamS;

        spatF_interpretS(0) = lranS;
        spatF_interpretS(1) = lsigU;
    }

}

void Hyperparameters::convert_interpret2theta(){

    // order: double ranS, double ranT, double sigU, double& lgamE, double& lgamS, double& lgamT
    if(dimList(1) > 0){

        double alpha_t = 1; 
        double alpha_s = 2;
        double alpha_e = 1;

        double alpha = alpha_e + alpha_s*(alpha_t - 0.5);
        //double nu_s  = alpha   - 1;
        double nu_s  = alpha - 1; 
        double nu_t  = alpha_t - 0.5; // because dim temporal domain always 1

        double ranS = spatTempF_interpretS(0);
        double ranT = spatTempF_interpretS(1);
        double sigU = spatTempF_interpretS(2);

        double lgamS = 0.5*log(8*nu_s) - ranS;
        double lgamT = ranT - 0.5*log(8*nu_t) + alpha_s * lgamS;

        double lgamE;

        if(manifold == "sphere"){
            double cR_t = std::tgamma(alpha_t - 1.0/2.0)/(std::tgamma(alpha_t)*pow(4*M_PI, 1.0/2.0));
            double cS = 0.0;
            double t_loop = - omp_get_wtime();
            for(int k=0; k<50; k++) // compute 1st 100 terms of infinite sum
            {  
                cS += (2.0*k + 1) / (4*M_PI* pow(pow(exp(lgamS), 2) + k*(k+1), alpha));
            }
            t_loop += omp_get_wtime();
            // printf("cS: %f\n", cS);
            //printf("sphere. c3 : %f, t loop : %f\n",  0.5*log(cR_t) + 0.5*log(cS), t_loop);
            lgamE = 0.5*log(cR_t) + 0.5*log(cS) - 0.5*lgamT - sigU;

        } else {
            //double c1_scaling_const = pow(4*M_PI, dim_spatial_domain/2.0) * pow(4*M_PI, 1.0/2.0); // second for temporal dim
            double c1_scaling_const = pow(4*M_PI, 1.5);
            //std::cout << "c1_scaling_const interpret2theta : " << c1_scaling_const << std::endl;
            double c1 = std::tgamma(nu_t)*std::tgamma(nu_s)/(std::tgamma(alpha_t)*std::tgamma(alpha)*c1_scaling_const);
            //printf("R^d. c3 : %f\n", 0.5*log(c1) - (alpha-dim_spatial_domain/2)*lgamS);
            lgamE = 0.5*log(c1) - 0.5*lgamT - (alpha-spatial_dim/2)*lgamS - sigU;
        }

        spatTempF_modelS(0) = lgamE; // lgamE
        spatTempF_modelS(1) = lgamS; // lgamS
        spatTempF_modelS(2) = lgamT; // lgamT

    }

    // order : double lranS, double lsigU, double& lgamE, double& lgamS
    if(dimList(2) > 0){
        // assuming alpha = 2
        double alpha    = 2;
        double nu_s  = alpha - spatial_dim/2.0;

        double lranS = spatF_interpretS(0);
        double lsigU = spatF_interpretS(1);

        double lgamS = 0.5*log(8*nu_s) - lranS;
        double lgamE = 0.5*(std::tgamma(nu_s) - (std::tgamma(alpha) + 0.5*spatial_dim*log(4*M_PI) + 2*nu_s*lgamS + 2*lsigU));
    
        spatF_modelS(0) = lgamE;
        spatF_modelS(1) = lgamS;
    }
}

#if 0 
// create structure to contain hyperparameters theta
struct Hyperparameters
{
    int spatial_dim = 0;
    std::string manifold = "";
    
    Vect lik; // log
    // equivalent -> _ms / _is : just in different parametrizations
    // TODO: function for conversion 
    Vect spatialF_ms;  // log hp related to spatial field in model scale. order:    
    Vect spatialF_is; // log hp related to spatial field in model scale. order: range s, sigma s

    Vect spTempF_ms;   // log hp related to spatial temporal field in model scale. order: 
    Vect spTempF_is;  // log hp related to spatial temporal field in model scale. order: range s, range t, sigma st
    
    Vector3i dimList = Vector3i::Zero();  // contains dim of each of the above: initialized to zero. update when above are being set.

};

#endif