library(INLA)

y <- rnorm(1)
xx <- rnorm(1)
print(c(y,xx))

mlik <- dnorm(xx, mean=0, sd=1, log=TRUE) +
  dnorm(y, mean=xx, sd=1, log=TRUE) -
  dnorm(xx, mean=y/2, sd=sqrt(1/2), log=TRUE)
print(mlik)

out_INLA = inla(y ~ 1, data =data.frame(y=y), family="stdnormal", control.fixed=list(prec.intercept=1))

print(out_INLA$mlik)


out_INLA = inla(y ~ 1, data =data.frame(y=y), family="gaussian", control.fixed=list(prec.intercept=1))
