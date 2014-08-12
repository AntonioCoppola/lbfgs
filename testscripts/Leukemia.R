# An example using L-BFGS and OWL-QN to perform logit and Poisson regressions 
# using data from Golub, Todd R., et al. "Molecular classification of cancer: 
# class discovery and class prediction by gene expression monitoring." 
# Science 286.5439 (1999): 531-537. A workspace with the dataset 
# ("Leukemia.RData") is included in the package distribution.

library(lbfgs)
library(glmnet)
library(microbenchmark)
data(Leukemia)

###########################
# Logit Examples (L-BFGS) #
###########################

# Standard logit setup
lhood <- function(par, X, y) {
  Xbeta <- X%*%par
  -sum(y*Xbeta - log(1 + exp(Xbeta)))
}

grad <- function(par, X, y) {
  p <-  1/(1 + exp(-X%*%par))
  -crossprod(X,(y-p))
}

# A version with a ridge penality

lhoodL2 <- function(par, X, y, prec) {
  Xbeta <- X%*%par
  -(sum(y*Xbeta - log(1 + exp(Xbeta))) -.5*sum(par^2*prec))
}

gradL2 <- function(par, X, y, prec) {
  p <-  1/(1 + exp(-X%*%par))
  -(crossprod(X,(y-p)) -par*prec)
}

init <- rep(0, ncol(X1))
X <- Leukemia$x
y <- Leukemia$y
X1 <- cbind(1, X)   # Add an intercept

# Comparing the ridge version to optim
optim.out <- optim(init, lhoodL2, gradL2, method = "L-BFGS-B", X=X1, y=y, prec=2)
lbfgs.out <-lbfgs(lhoodL2, gradL2, init, invisible=1, X=X1, y=y, prec=2)
all.equal(optim.out$value, lbfgs.out$value)
microbenchmark(optim.out <- optim(init, lhoodL2, gradL2, method = "L-BFGS-B", X=X1, y=y, prec=2),
               lbfgs.out <- lbfgs(lhoodL2, gradL2, init, invisible=1, X=X1, y=y, prec=2))


#############################
# Poisson Examples (OWL-QN) #
#############################

mod <- glmnet(X, y, family=c("binomial"))
coef(mod)[,50] # Answer
mod$lambda[50] # Regularization parameter

prec <- 0 # Setting the precision to 0 when doing L1, since the penalty is under orthantwise_c

out <- lbfgs(lhoodL2, gradL2, rep(1, ncol(X1)), invisible=1, orthantwise_c=1, 
                         linesearch_algorithm="LBFGS_LINESEARCH_BACKTRACKING",
                         orthantwise_start = 1,
                         orthantwise_end = ncol(X1),
                         X=X1, y=y, prec=0))


all.equal(which(coef(mod)[,70]!=0), which(owl$par!=0))

# Poisson regression setup
pois.ll <- function(par, X, y, prec=0) {
  Xbeta <- X%*%par
  -(sum(y*Xbeta - exp(Xbeta)) -.5*sum(par^2*prec))
}

pois.grad <- function(par, X, y, prec=0) {
  expXbeta <- exp(X%*%par)
  -(crossprod(X,(y-expXbeta)) -par*prec)
}

counts <- c(18,17,15,20,10,20,25,13,12)
outcome <- gl(3,1,9)
treatment <- gl(3,3)
print(d.AD <- data.frame(treatment, outcome, counts))
glm.D93 <- glm(counts ~ outcome + treatment, family = poisson(), x=TRUE, y=TRUE)
X <- glm.D93$x
y <- counts

# Double-checking the coding
eval <- rnorm(ncol(X))
library(numDeriv)
numeric <- numDeriv::grad(pois.ll, x=eval, X=X, y=y, prec=0)
analytic <- pois.grad(eval, X=X, y=y, prec=0)
all.equal(numeric,as.numeric(analytic))

optim.out <- optim(rep(0,5), pois.ll, pois.grad, X=X, y=y, method="L-BFGS-B")
cbind(as.numeric(coef(glm.D93)),optim.out$par) # Minor differences due to convergence tolerance differences


# Poisson example from glmnet
N <- 500; p <- 20
nzc <- 5
x <- matrix(rnorm(N*p),N,p)
beta <- rnorm(nzc)
f <- x[,seq(nzc)]%*%beta
mu <- exp(f)
y <- rpois(N,mu)
fit <- glmnet(x,y,family="poisson", standardize=FALSE)
C <- fit$lambda[25]*nrow(x)
coef(fit)[,25]
library(RlibLBFGS)
X <- x
X1 <- cbind(1,X)


owl <- lbfgs(pois.ll,pois.grad, rep(0, ncol(X1)), X=X1, y=y, prec=0, 
                           invisible=1, orthantwise_c=C, 
                           linesearch_algorithm="LBFGS_LINESEARCH_BACKTRACKING",
                           orthantwise_start = 1,
                           orthantwise_end = ncol(X1))

cbind(owl$par,as.numeric(coef(fit)[,25]))

# Timings
microbenchmark(glmnet(x,y,family="poisson", standardize=FALSE),
               owl <- lbfgs(pois.ll,pois.grad, rep(0, ncol(X1)), invisible=1, orthantwise_c=C, 
                                    linesearch_algorithm="LBFGS_LINESEARCH_BACKTRACKING",
                                    orthantwise_start = 1,
                                    orthantwise_end = ncol(X1),
                                    X=X1, y=y, prec=0))
