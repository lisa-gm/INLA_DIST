## precision model matrices

########## function to assemble submatrices ###########

### function to build Q.k(g.2) spatial matrix
Qgk.fun <- function(fem, g=1, order=3) {
  if (order==1)
    return(g^2 * fem$c0 + fem$g1)
  if (order==2)
    return(g^4 * fem$c0 + 2 * g^2 * fem$g1 + fem$g2)
  if (order==3)
    return(g^6 * fem$c0 + 3 * g^4 * fem$g1 +
             3 * g^2 * fem$g2 + fem$g3)
  return(NULL)
}

Qs.fun <- function(sfem, g, order=2){
  Qs <- g[1]^2*Qgk.fun(sfem, g[2], order)  
  return(Qs)
}


########## function to assemble overall spatial precision matrix ##########
# INLA only
### practical range = sqrt(8*nu)/g[2]  (but nu is one because order = 2) -> practical range is a distance. rho(prac range) \approx 0.13 = correlation
### assemble Q.u
Qst.fun <- function(sfem, tfem, g) {
  q1s <- Qgk.fun(sfem, g[2], 1)
  #print("q1s")
  #print(q1s[1:10,1:10])
  q2s <- Qgk.fun(sfem, g[2], 2)
  #print("q2s")
  #print(q2s[1:10,1:10])
  q3s <- Qgk.fun(sfem, g[2], 3)
  #print("q3s")
  #print(q3s[1:10,1:10])
  
  M0 <- tfem$c0
  #print("M0")
  #print(M0[1:10,1:10])
  nt <- nrow(M0)
  M1 <- sparseMatrix(
    i=c(1, nt),
    j=c(1, nt),
    x=c(0.5, 0.5)) 
  #print("M1")
  #print(M1[1:10,1:10])
  M2 <- tfem$g1
  #print("M2")
  #print(M2[1:10,1:10])
  Qst = g[1]^2*(kronecker(M0, q3s) +
                kronecker(M1 * g[3], q2s) +
                kronecker(M2 * g[3]^2, q1s))
  return(Qst) 
}


Qprior.fun <- function(sfem, tfem, g, nb, prior_FE=1e-3){
  Qst <- Qst.fun(sfem, tfem, g)
  nu = Qst@Dim[1]

  zeroMat   <- sparseMatrix(i = integer(0), j = integer(0), dims = c(nu, nb))
  precFEMat <- Diagonal(x = rep(prior_FE, nb))

  Qprior <- rbind(cbind(Qst, zeroMat), cbind(t(zeroMat), precFEMat))
}