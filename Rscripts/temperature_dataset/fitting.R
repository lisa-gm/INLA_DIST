write_to_file = FALSE #FALSE
call_INLA = TRUE  #FALSE

mesh_scale <- 150 ### increase spatial resolution lowering this
t.resol <- 1 ## to define temporal mesh resolution (one way to lower the time resolution)

### packages
library(INLA)
library(INLAspacetime)
library(inlabru)

if (any(Sys.info()['user']==c('elias', 'eliask', 'krainset'))) {
    file_path <- here::here("rcode", "application")
    write_to_file = FALSE
    call_INLA <- TRUE
    if(any(Sys.info()['user']==c('elias', 'eliask'))) 
        inla.setOption(inla.call='remote')
} else {
    #file_path <- "/home/x_gaedkelb/b_INLA/data/temperature"
    #file_path <- "/home/x_gaedkelb/pardiso_st_spde/rcode/application" #b_INLA/data/temperature"
    file_path <- getwd()
}

fit.fl <- file.path(file_path, "fit.rds")

stations <- readRDS(file.path(file_path, "stations.rds"))
wtavg <- readRDS(file.path(file_path, "wtavg.rds"))

if(FALSE) ### time basis fn
    plot(sin, 0, pi)

# nt hardcoded to 365
nt = 365

### create the data
iis <- which(!is.na(wtavg))
ldata <- data.frame(
    beta0 = 1,
    elevation = rep(stations$elevation/1000, 365),
    xloc = rep(coordinates(stations)[, 1], 365),
    yloc = rep(coordinates(stations)[, 2], 365), 
    time = rep(1:365, each = nrow(wtavg)),
    tavg = as.vector(wtavg))[iis, ]
ldata$stime <- sin(pi * ldata$time/max(ldata$time))

range(ldata$yloc)
ldata$north <- (ldata$yloc - 3000) / 1000 

print(str(ldata))

(ndata <- nrow(ldata))

bound <- inla.nonconvex.hull(
    coordinates(stations),
    max(10, mesh_scale),
    resolution = 100)

smesh <- inla.mesh.2d(
    boundary = bound, 
    max.edge = c(2, 10) * mesh_scale, 
    offset = c(1e-7, 50*mesh_scale),
    cutoff = mesh_scale/2,
    min.angle = 25)

(ns <- smesh$n)

if(FALSE) {

    plot(smesh, asp=1)
    points(stations, pch=19, col=2)

}

tmesh <- inla.mesh.1d(seq(1,365+t.resol*.9,t.resol))

(nt <- tmesh$n)

cat("s =", ns, "t =", nt,
    "st =", ns * nt,
    "ndata = ", ndata, "\n")

var(ldata$tavg, na.rm = TRUE)
sd(ldata$tavg, na.rm = TRUE)
summary(ldata$tavg)

### define the spatial Finite Element Matrices
sfem <- inla.mesh.fem(smesh, order=3)

### define the temporal Finite Element Matrices
tfem <- inla.mesh.fem(tmesh, order=2)

##############################################################################
# generate necessary matrices to write to file for C++ code
# spatial temporal model write base matrices to file
if(write_to_file && (nt > 1)){

    B <- cbind(beta0=1, as.matrix(ldata[c("elevation", "stime", "north")]))
    nb <- ncol(B)
    
    n = ns*nt+nb
                                        # create folder for files
    dir_name = paste0("ns", toString(ns),
                      "_nt", toString(nt), "_nb", toString(nb))
    base_path <- file.path(file_path, dir_name) 

    A.st <- inla.spde.make.A(
        mesh = smesh,
        loc = as.matrix(ldata[c("xloc", "yloc")]),
        group = ldata$time,
        group.mesh = tmesh)
    
    A.x <- cbind(A.st, B)

    print("dim(A.x)")
    print(dim(A.x))

    # source files with write functions           
    source(file.path("../../rcode/fun_write_file.R"))
      
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
    mat_to_file_sp.fun(A.x, "Ax", base_path)

    print(paste("dim(y) = ", ndata))
    mat_to_file.fun(ldata$tavg, "y", base_path)

}

if(call_INLA){
    
if(file.exists(fit.fl)) {

    fit <- readRDS(fit.fl)
    
} else {

    ws <- sfem$c0@x/sum(sfem$c0@x)
    wt <- tfem$c0@x/sum(tfem$c0@x)
    stConstr <- list(
        A=matrix(kronecker(wt, ws), nrow = 1), e=0)
    stConstr$A[1, ] <- stConstr$A[1,] / sum(stConstr$A[1,])
    
    stmodel <- stModel.define(
        smesh = smesh, 
        tmesh = tmesh,
        model = "121",
        control.priors=list(
            prs=c(300, 0.01),
            prt=c(1, 0.01),
            psigma=c(5, 0.01)))
    
### print number of non-zeros in Q_u
    cat("Number of non-zeros in Q_u:",
        stmodel$f$cgeneric$data$matrices$xx[2], "\n")
    
### define the mapper for the spacetime model
    stmapper <- bru_get_mapper(stmodel)
    
### define model components
    Mcomps <- ~ 0 + mu(1) + elevation + stime + north + 
        spacetime(list(space = cbind(xloc, yloc), time=time),
                  model=stmodel, mapper=stmapper,
                  extraconstr=stConstr)
    
### set prior for the likelihood parameter
    pprec <- list(theta=list(prior="pc.prec", param=c(5, 0.01)))
    
### the likelihood part
    lhood <- inlabru::like(
        formula = tavg ~ .,
        family="gaussian",
        control.family = list(
            hyper = pprec),
        data=ldata)
    
    # TODO: check why doesn't work.
    #cat(summary(component_list(
    #    Mcomps, lhoods=list(lhood), verbose=FALSE)))
    
    inla.setOption(
        inla.mode = "compact", 
        smtp = "pardiso",
        pardiso.license = "~/.pardiso.lic")
    
    ccomp <- list(dic = TRUE, waic = TRUE, cpo = TRUE)
    cinla <- list(parallel.linesearch = TRUE, 
                  strategy = "gaussian", 
                  int.strategy = "eb", 
                  control.vb = list(enable = FALSE))
    
    cat("Fitting the model.\n")
    t0 <- Sys.time()

    theta.ini <- c(-1, ## tau_e
                   7, 3, 2) ## rs, rt, sigma
    
    fit <- bru(
        components = Mcomps,
        lhood,
        options=list(
            verbose = TRUE, 
            control.compute = ccomp,
	    control.inla = cinla,
            control.mode = list(theta = theta.ini, restart = TRUE),
            num.threads = "8:12", ### manage it accordly to available resource
            safe = FALSE
        )
    )

    cat("Fit done, ")
    print(Sys.time() - t0)
    
### select elements of the output to save
    rnams <- c(paste0("summary.",
                      c("fixed", "hyperpar", "random", "fitted.values")),
               "internal.marginals.hyperpar", "mlik", "mode", "cpu.used")
    
    sres <- fit[rnams]
    sres$summary.fitted.values <-
        fit$summary.fitted.values[1:ndata, c("mean", "sd")]
    rownames(sres$summary.fitted.values) <- NULL ### to use less mem
    
    saveRDS(sres, fit.fl)

}

if(write_to_file && (nt > 1)) {
    
    cat(unname(fit$mode$theta),
        file = file.path(
            base_path,
            "mode_theta_interpret_param_INLA.txt"))              
    
    write(x = fit$misc$cov.intern,
          file = file.path(
              base_path,
              "inv_hessian_mode_theta_interpret_param_INLA.txt"),
          ncolumns = 4)
    
    write(x = cbind(1:ncol(A.x),
                    c(fit$summary.fixed$mean,
                      fit$summary.fixed$random$spacetime$mean)),
          file = file.path(
              base_path,
              "mean_latent_parameters_INLA.txt"), 
          ncolumns = 2)
    
    write(x = cbind(1:ncol(A.x),
                    c(fit$summary.fixed$sd,
                      fit$summary.fixed$random$spacetime$sd)),
          file = file.path(
              base_path,
              "sd_latent_parameters_INLA.txt"), 
          ncolumns = 2)
    
}

} # end call_INLA
