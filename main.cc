#include <armadillo>




struct DATA
{	
	int no;
	arma::sp_mat A_st;
	arma::mat B;
	arma::vec y;
	double yTy;
};



struct MODEL
{
	int ns;
	int nt;
	int nb;
	arma::sp_mat q1s;
	arma::sp_mat q2s;
	arma::sp_mat q3s;
	arma::sp_mat M0 ;
	arma::sp_mat M1;
	arma::sp_mat M2;
	arma::sp_mat Q_b;
};






void likelihood(arma::vec theta){}


void log_det(arma::vec theta){
  



  // RGF<T> *solver;
  // solver = new RGF<T>(ia, ja, a, ns, nt, nb);

}

arma::sp_mat construct_Qu(arma::vec theta){

}

void cond(arma::vec theta){}


// primary function
void post_th(arma::vec theta, double func_val, arma::vec mu, struct *DATA, struct *MODEL){


	arma::sp_mat Qu = construct_Qu(theta, )


}



void main(){

	// construtct MODEl struct
	MODEL model{};

	// read file for MODEL
	model.ns = atoi(argv[2]);
	model.nt = atoi(argv[3]);
	model.nb = atoi(argv[4]);
	model.q1s = readCSC_sym( );
	model.q2s = readCSC_sym( );
	model.q3s = readCSC_sym( );
	model.M0  = readCSC_sym( );
	model.M1 = readCSC_sym( );
	model.M2 = readCSC_sym( );
	model.Q_b = readCSC_sym( );

	DATA data{};

	data.no = atoi(argv[5]);
	data.A_st = readCSC();();
	data.B = read_matrix();
	data.y = read_matirx();

	data.yTy; //t(y)*y;

	


}



