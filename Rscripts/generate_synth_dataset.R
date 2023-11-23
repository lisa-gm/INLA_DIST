### BFGS spacetime model
rm(list = ls())  ### clear environment

library("INLA")
library("optparse")
library("mosaic")

option_list = list(
    #write_to_file 
    make_option(c("-w", "--writeToFile"), 
        default=TRUE, type="logical", help="write matrices to file", metavar ="logical"),
    #call_INLA
    make_option(c("-i", "--callINLA"), 
        default=TRUE, type="logical", help="call INLA", metavar ="logical"),
    # sresolution    
    make_option(c("-s", "--spatialGridSize"), 
        default=5, type="integer", help="choose spatial grid size which will be squared", metavar ="number"),
    # number of timesteps
    make_option(c("-t", "--temporalGridSize"), 
        default=10, type="integer", help="choose temporal grid size", metavar ="number"),    
    # number of fixed effects
    make_option(c("-b", "--NoFixedEffects"), 
        default=6, type="integer", help="choose between 2,4 and 6", metavar ="number"),
    make_option(c("-p", "--pardisoLic"), 
        default=FALSE, type="logical", help="create plots", metavar ="logical"),
    make_option(c("-c", "--constraints"),
        default=FALSE, type="logical", help = "disable or enable sum-to-zero constraints in INLA.", metavar="logical"),
    make_option(c("-r", "--saveRobject"),
        default=FALSE, type="logical", help = "save matrices to R object", metavar="logical")
    # no_obs
    #make_option(c("-o", "--observations"), 
    #    default=, type="integer", help="choose grid size", metavar ="number")
    # output file
    #make_option(c("-l", "--log"), 
    #    default="out.txt", type="character", help="output file name [default= %default]", metavar="character")
); 

# example call (if not using default parameters) :
# Rscript generate_synth_dataset.R -w F -i T -s 2 -t 3 -b 2 
# Rscript generate_synth_dataset.R -w T -i T -s 5 -t 10 -b 6 -p F
# Rscript generate_synth_dataset.R -w T -i F -s 3 -t 5 -b 4 -p F
# Rscript generate_synth_dataset.R -w F -i T -s 10 -t 16 -b 6 -p F -c T
# Rscript generate_synth_dataset.R -w T -i F -s 32 -t 30 -b 6 -p F -c F
# Rscript generate_synth_dataset.R -w T -i F -s 40 -t 30 -b 6 -p F -c F
# Rscript generate_synth_dataset.R -w T -i F -s 44 -t 30 -b 6 -p F -c F


opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

write_to_file = opt$writeToFile
call_INLA     = opt$callINLA
sresolution   = opt$spatialGrid
nt            = opt$temporalGridSize
nb            = opt$NoFixedEffects
pardisoLic    = opt$pardisoLic
constr        = opt$constraints
save_object   = opt$saveRobject

data_type = "synthetic"

############# set paths ##################

base_path = paste0("../data/", data_type)

#############################################################

### create a mesh over the globe for a given resolution
gmesh <- inla.mesh.create(globe=sresolution)
### number of nodes in the spatial mesh
ns <- gmesh$n
ns

### define the spatial Finite Element Matrices
sfem <- inla.mesh.fem(gmesh, order=3)

### define the temporal mesh for a given number of times
tmesh <- inla.mesh.1d(loc=1:nt)
### number of nodes in the temporal mesh
nt <- tmesh$n 
nt

### define the temporal Finite Element Matrices
tfem <- inla.mesh.fem(tmesh, order=2)

### number of spatial locations where data is observed, 
# at each time point we then have 1 observation
stations = floor(2*ns)
print("number of samples per time step : ")
print(stations)
no = nt * stations
t_list = 1:nt

##########################################################################################################
# generate SPATIAL TEMPORAL data set 

### reasonable parameter values
interpretable.parameters <- c(
sigma.e=0.5, ### noise (nugget effect)
sigma.u=4, ### signal std
#sigma.u=3, ### signal std
rs=1, ### spatial correlation in the sphere decay 
rt=10) ### temporal correlation decay length

# log precision observations, log(gamma_E, gamma_s, gamma_t) 
# -> gamma_E^2*(kron(M0, q3s(gamma_s)) + 2*gamma_t*kron(M1, q2s(gamma_s)) + gamma_t^2*kron(M2, q1s(gamma_s)))
source("parametrisation_hyperparam.R", chdir=T)
# why do we not do log for sigma.e but log(1/sigma.e^2)
#l_interpret_param <- log(interpretable.parameters) 
#theta.original <- c(log(1/interpretable.parameters[[1]]^2), interpret2theta(log(interpretable.parameters[2:4])))

l_interpret_param <- c(log(1/interpretable.parameters[1]^2), log(interpretable.parameters[2:4]))
print("l_interpret_param")
print(l_interpret_param)

theta.original <- c(l_interpret_param[1],
                  interpret2theta(log(interpretable.parameters[2:4])))
print("theta original")
print(theta.original)


########### initialise FIXED EFFECTS ##############
if(nb == 2){
    print("CHOOSE MORE FIXED EFFECTS OR FIX CODE.")
    beta0 = -1
    beta1 = 3
    beta = rbind(beta0, beta1)

} else if(nb == 4){
    beta0 = -1
    beta1 = 3
    beta2 = 0.5
    beta3 = 2
    beta = rbind(beta0, beta1, beta2, beta3)

} else if(nb == 6){
    beta0 = -1
    beta1 = 3
    beta2 = 0.5
    beta3 = 2
    beta4 = 1
    beta5 = -2
    beta = rbind(beta0, beta1, beta2, beta3, beta4, beta5)

} else {

    print("invalid number of fixed effects nb. Choose 2,4 or 6!")
    exit()
}


#### randomly sample locations in latitude-longitude for the observations
latlong = rgeo(stations)
longlat = latlong[,2:1]

### map the data locations (in longlat) to the sphere
# rename spherical coordinates
coo.sphere <- inla.mesh.map(
  longlat, 'longlat', inverse=TRUE)

dim(A.st <- inla.spde.make.A(gmesh, kronecker(matrix(1,nt),
                                             coo.sphere),
                            group=rep(t_list, each=nrow(coo.sphere)),
                            group.mesh=tmesh))


# require Q.u to sample from solution
### construct spatial-temporal precision matrix
source("spatial_temporal_model_matrices.R", chdir=T)   
Q.u <- Qst.fun(sfem, tfem, exp(theta.original[2:4])) 

# if seed is set : slow but reproduceable
system.time(u <- drop(inla.qsample(n=1, Q=Q.u, seed=7)))
#system.time(u <- drop(inla.qsample(n=1, Q=Q.u)))
#print(u[1:10])
    
print("summary(u)")   
summary(u)

# center u
u = mean(u) - u

print("summary(u) after centering")   
summary(u)

if(FALSE)
    plot(drop(u)) ### does behaves strange over time with strange parameters...

# create covariates 
set.seed(3)

# introduce covariates that have periodic trends in time steps : sine (do one full period)
t_sin = rep(sin((2*pi)/nt*t_list), each = stations) + rnorm(no, sd=0.5)
#plot(t_sin)

if(nb == 2){
    # include offset
    B <- cbind(rep(1, no), t_sin)
} else {

    # simply sample everything from standard normal distribution
    rnd = matrix(data=rnorm(no*(nb-2)), ncol=(nb-2))

    B <- cbind(rep(1, no), rnd, t_sin)
}

# what else could there be?


print("dim B: ")
print(dim(B))

eta = B %*% beta + A.st %*% u

# draw k samples for each spatial node
y = rnorm(no, mean = drop(eta), sd=exp(-theta.original[1]/2))

print("summary(y)")   
summary(y)

A.x <- cbind(A.st, B)
#A.x <- B

# spatial temporal model write base matrices to file
if(write_to_file == TRUE && nt > 1){

    # source files with write functions           
    source("fun_write_file.R", chdir=T)

    # create folder for files
    dir_name = paste0("ns", toString(ns), "_nt", toString(nt), "_nb", toString(nb))
    base_path <- file.path(base_path, dir_name) 
    dir.create(base_path)

    # spatial matrices
    c0 = as(sfem$c0, "dgCMatrix")
    mat_to_file_sym.fun(c0, "c0", base_path)
    g1 = as(sfem$g1, "dgCMatrix")
    mat_to_file_sym.fun(g1, "g1", base_path)
    g2 = as(sfem$g2, "dgCMatrix")
    mat_to_file_sym.fun(g2, "g2", base_path)
    g3 = as(sfem$g3, "dgCMatrix")
    mat_to_file_sym.fun(g3, "g3", base_path)

    # temporal matrices
    # convert to dgCMatrix format
    M0_ddi <- tfem$c0
    #M0 <- sparseMatrix(i=c(1:M0_ddi@Dim[1]),j=c(1:M0_ddi@Dim[2]),x=M0_ddi@x,dims=list(M0_ddi@Dim[1],M0_ddi@Dim[2]))
    M0 = as(tfem$c0, "dgCMatrix")
    mat_to_file_sym.fun(M0, "M0", base_path)
    M1 <- sparseMatrix(
      i=c(1, nt),
      j=c(1, nt),
      x=c(0.5, 0.5)) 
    mat_to_file_sym.fun(M1, "M1", base_path)
    M2 <- tfem$g1
    mat_to_file_sym.fun(M2, "M2", base_path)

    # store A.x and not A.st & B separately
    mat_to_file_sp.fun(A.x, "Ax", base_path, no_obs=no)
    mat_to_file.fun(y, "y", base_path, no_obs=no)

    # also write original parameters to file
    # ns, nt, nb, no, interpretable.parameters, theta.original, fixed.effects
    writeLines(
        c(paste("ns", ns),
          paste("nt", nt),
          paste("nb", nb),
          paste("no", no),
          paste("interpret.param", paste(round(l_interpret_param, digits = 3), collapse = " ")),          
          paste("theta.original ", paste(round(theta.original,    digits = 3), collapse = " ")),  
          paste("fixed effects  ", paste(round(beta,              digits = 3), collapse = " "))            
          ), 
        file.path(base_path, "true_parameters.txt"))

} else if(write_to_file == TRUE && nt == 1){

        # source files with write functions           
    source("fun_write_file.R", chdir=T)

    # create folder for files
    dir_name = paste0("ns", toString(ns), "_nt", toString(nt), "_nb", toString(nb))
    base_path <- file.path(base_path, dir_name) 
    dir.create(base_path)

    # spatial matrices
    c0 = as(sfem$c0, "dgCMatrix")
    mat_to_file_sym.fun(c0, "c0", base_path)
    g1 = as(sfem$g1, "dgCMatrix")
    mat_to_file_sym.fun(g1, "g1", base_path)
    g2 = as(sfem$g2, "dgCMatrix")
    mat_to_file_sym.fun(g2, "g2", base_path)

    # store A.x and not A.st & B separately
    mat_to_file_sp.fun(A.x, "Ax", base_path, no_obs=no)
    mat_to_file.fun(y, "y", base_path, no_obs=no)

    # also write original parameters to file
    # ns, nt, nb, no, interpretable.parameters, theta.original, fixed.effects
    writeLines(
        c(paste("ns", ns),
          paste("nt", nt),
          paste("nb", nb),
          paste("no", no),
          paste("interpret.param", paste(round(l_interpret_param, digits = 3), collapse = " ")),          
          paste("theta.original ", paste(round(theta.original,    digits = 3), collapse = " ")),  
          paste("fixed effects  ", paste(round(beta,              digits = 3), collapse = " "))            
          ), 
        file.path(base_path, "true_parameters.txt"))

}

## save files to such that they can be reloaded with R
if(save_object){
    print("saving objects ...")
    file_name = file.path("synth_dataset", "RData_files", paste0("fixed_synth_dataset_ns" ,ns, "_nt", nt, "_nb", nb, ".RData"))
    save(gmesh, tmesh, y, B, A.st, file = file_name)
}


if(FALSE) { ### visualize the signal

### organize the signal (sampled) into a matrix
    u.s.t <- matrix(drop(u), gmesh$n)
    dim(u.s.t)
    
    gmesh.moll <- inla.mesh.map(
        gmesh$loc, 'mollweide', inverse=FALSE)

    real_to_ab <- function(x, a=0, b=1, aa=min(x), bb=max(x)) {
        return(a + (b-a) * (x - aa)/(bb-aa))
    }

### plot it for some of the time
    par(mfrow=c(4,4), mar=c(0,0,0,0))
    for (k in 1:16) {
        uuk <- real_to_ab(u.s.t[,k], aa=min(u), bb=max(u))
        ucol <- rgb(uuk, 1-2*abs(uuk-0.5), 1-uuk)
        plot(gmesh.moll, pch=19, cex=2, col=ucol,
             axes=FALSE, xlab='', ylab='')
    }

### projector to a grid (longlat)
    nxy <- c(400,200)
    ggproj <- inla.mesh.projector(gmesh, dims=nxy)
    st.grid <- inla.mesh.project(ggproj, u.s.t)

    dim(st.grid)
    qq <- quantile(as.vector(st.grid), 0:100/100, na.rm=TRUE)

    kk <- round(seq(1, nt, length=16))
    bb <- matrix(c(-180, -90, 180, 90), 2)

    file_name = file.path(source_path, paste0("synthetic_globe_ns", gmesh$n, "_nt", nt, "nb", nb, ".png"))
    file_name
    
      png(file=file_name,width=1000, height=2000)

      mfr <- c(min(length(kk), ceiling(sqrt(length(kk))*1.0)), NA)
      mfr[2] <- ceiling(length(kk)/mfr[1])
      if(((mfr[1]-1)*mfr[2])>=length(kk)) mfr[1] <- mfr[1]-1
      mfr
      
      par(mfrow=mfr, mar=c(0,0,0,0))
      for(k in kk) {
          image(ggproj$x, ggproj$y,
                matrix(st.grid[,k], nxy[1]), 
                xlab='', ylab='', axes=FALSE, asp=1,
                col=hcl.colors(100), breaks=qq,
                xlim=bb[1,]*c(1, ifelse(k==kk[length(kk)],1.2,1)), ylim=bb[2,])
          #legend('topleft', '', title=k, bty='n')
          map(add=TRUE)
      }
      image(c(bb[1,2], bb[1,2]*1.1),
            seq(bb[2,1], bb[2,2], length=100),
            matrix(0.5*(qq[1:100]+qq[2:101]), 1),
            breaks=qq, col=hcl.colors(100), add=TRUE)
      text(rep(bb[1,2]*1.15, 6),
           seq(bb[2,1], bb[2,2], length=6),
           format(qq[c(1, 20, 40, 60, 80, 101)], digits=2))
        
      dev.off()

    
}

##########################################################################################################
# CALL INLA


if(call_INLA == TRUE){

    if(nb == 2){
        sdat <- inla.stack(
          tag='data',
          data=list(y=y),    
          effects=list(data.frame(b0=1, b1=B[,2]),
                       list(s=1:ncol(A.st))),
          A=list(1, A.st))
    
    } else if(nb == 4){
        sdat <- inla.stack(
          tag='data',
          data=list(y=y),    
          effects=list(data.frame(b0=1, b1=B[,2], b2=B[,3], b3=B[,4]),
                       list(s=1:ncol(A.st))),
          A=list(1, A.st))

    } else{
        sdat <- inla.stack(
          tag='data',
          data=list(y=y),    
          effects=list(data.frame(b0=1, b1=B[,2], b2=B[,3], b3=B[,4], b4=B[,5], b5=B[,6]),
                       list(s=1:ncol(A.st))),
          A=list(1, A.st))
    }


### new: cgeneric
    
    library(INLAspacetime)
    
    stmodel <- stModel.define(
        smesh = gmesh,
        tmesh = tmesh,
        model = "121",
        constr = constr,
        control.priors = list(
            prs = c(0.1, 0.01),
            prt = c(1, 0.01),
            psigma = c(1, 0.01))
    )
    
    if(constr)
        print("sum-to-zero constraints enabled!")        
    
    if(nb == 2){
### define the model formula (with b0 explicit)
        formulae <- y ~ 0 + b0 + b1 +
            f(s, model=stmodel)
    } else if(nb == 4){
        formulae <- y ~ 0 + b0 + b1 + b2 + b3 +
            f(s, model=stmodel) 
    } else {
        formulae <- y ~ 0 + b0 + b1 + b2 + b3 + b4 + b5 +
            f(s, model=stmodel)
    }
    
    if(pardisoLic){
        print("checking for PARDISO license.")
        ### PARDISO setup
        inla.setOption(
        ##inla.timeout=1   #limit runtime -> then have to average over displayed times 
        # -> check if they roughly align with total time ...
        pardiso.license='~/pardiso.lic',
        smtp = "pardiso",
        num.threads="9:4")  # can't pass in variables ... try 8:8, 4:16, 9:4
    }

    ### fit the model
    print(interpretable.parameters)
    print(theta.original)

    ## start with the same initial guess ...

### the cgeneric model consider the order log(r_s, r_t, sigma_u) 
    ini.theta2i <- log(interpretable.parameters[c(3, 4, 2)]) 
    c(theta.original[1], ini.theta2i)
    
    ## make sure to consider the same initial guess as in my version
    # c(noise obs, lgamE, lgamS, lgamT)
    # init.theta <- c(2, -3, 1.5, 5)
    ## c(noise obs, ranT, ranS, sigU)
 ### order in cgeneric: log(rs, rt, sigma)
    initial.interpret.theta <- c(2.000000, -0.460279,  2.693147, -2.612086)
    #initial.interpret.theta <- c(4,0,0,0)

    ##initial.interpret.theta = c(theta.original[1]+1,ini.theta2i+c(1,1,-1))
    print("initial theta : " , initial.interpret.theta)
    
    # require more recent inla version for : inla.mode="experimental" ?!
    out <- inla(formulae,
                data=inla.stack.data(sdat),
                control.predictor=list(
                  A=inla.stack.A(sdat),
                  compute=TRUE),
                verbose=TRUE,
                inla.mode="compact",
                control.mode=list(
                    theta=initial.interpret.theta,
                    restart=TRUE), 
                control.inla=list(int.strategy='eb',
                                  parallel.linesearch=FALSE #,
                                  #tolerance=1e-8 
                                  ) #,
                ##control.compute = list(config = TRUE),keep=TRUE
                )
                
    tail(out$logfile, 15)
    out$cpu

    # config=TRUE, keep=TRUE
    #res=inla.collect.results("/home/x_gaedkelb/pardiso_st_spde/rcode/prototype_INLA/inla.model/results.files")
    #print(res)

    ### the log of the model hyper-parameters
    print("model parameter scale")
    print(rbind(original=c(theta.original), 
          output=c(out$mode$theta[1], interpret2theta(out$mode$theta[c(4,2,3)]))))

    print("interpretable parameter scale")
    print(rbind(original=interpretable.parameters,
          c(exp(-out$mode$theta[1]/2), exp(out$mode$theta[c(4,2,3)]))))

    if(FALSE){
        ### posterior mean vs simulated
        plot(out$summary.ran[[1]]$mean, u, asp=1)
        abline(0:1, col=2)
    }

    ### user scale model parameters posterior summary
    print(cbind(true=c(tau.e=1/interpretable.parameters[1]^2,
                 ini.theta2i),   out$summary.hyperpar))

    # reverse order and obtain original parametrisation
    print("estimated theta")
    print(c(out$mode$theta[1], interpret2theta(out$mode$theta[c(4,2,3)])))
    print("original theta : ")
    print(theta.original)

    print("covariance : ")
    print(out$misc$cov.intern)

    print("fixed effects")
    print(cbind(original=beta,out$summary.fixed))

    print("random effects mean")
    print(out$summary.random$s$mean[1:20])
    print("random effects sd[1:20]")
    print(out$summary.random$s$sd[1:20])

} # end if call_INLA == true




