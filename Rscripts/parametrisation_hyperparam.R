# change between model parametrisations

### function to map from SPDE to interpretable parameters 
###  from log(gamma_E, gamma_s, gamma_t) 
# -> gamma_E ^ 2 (kron(M0, q3s(gamma_s)) + 2*gamma_t*kron(M1, qs2(gamma_s)) + gamma_t^2*kron)
###  to log(sigma, range_s, range_t) 
theta2interpret <- function(theta) {
    alpha_t = 1; alpha_s = 2; alpha_e = 1
    alpha = alpha_e + alpha_s*(alpha_t-1/2)
    nu.s = alpha -1; nu.t = alpha_t - 0.5
    c1 = gamma(alpha_t - 1/2)*gamma(alpha-1)/(gamma(alpha_t) *gamma(alpha) *8*pi^1.5)
    gE <- exp(theta[1])
    gs <- exp(theta[2])
    gt <- exp(theta[3])
    rt <- gt*sqrt(8*(alpha_t -0.5))/(gs^alpha_s)
    rs <- sqrt(8*nu.s)/gs
    sigma <- sqrt(c1)/(gE*sqrt(gt)*(gs^(alpha-1)))
    return(log(c(sigma, rs, rt)))
}

### function to map from interpretable to the SPDE parameters 
###  from log(sigma, range_s, range_t) 
# -> gamma_E^2*(kron(M0, q3s(gamma_s)) + 2*gamma_t*kron(M1, q2s(gamma_s)) + gamma_t^2*kron(M2, q1s(gamma_s)))
###  to log(gamma_E, gamma_s, gamma_t) 
# th2 = gam_ E, th3 = gam_s, th4=gam_t
interpret2theta <- function(theta.interpret) {
### Note that theta is log(sigma, range_s, range_t)
        alpha_t = 1; alpha_s = 2; alpha_e = 1
        alpha = alpha_e + alpha_s*(alpha_t-1/2)
        nu.s = alpha -1; nu.t = alpha_t - 0.5
    c1 = gamma(alpha_t - 1/2)*gamma(alpha-1)/(gamma(alpha_t) *gamma(alpha) *8*pi^1.5)
### Define the log-gamma's
        theta.gam = rep(NA, 3)
        theta.gam[2] = 0.5*log(8*nu.s) - theta.interpret[2]
        theta.gam[3] = theta.interpret[3] - 0.5*log(8*(alpha_t-1/2)) + alpha_s * theta.gam[2]
        theta.gam[1] = 0.5*log(c1) - 0.5*theta.gam[3] - (alpha-1)*theta.gam[2] - theta.interpret[1]
        return(theta.gam) ### log(gamma_E, gamma_s, gamma_t)
}