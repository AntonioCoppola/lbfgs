# Leukemia Data
# Golub et al 1999
# binary response

library(lbfgs)
load("Leukemia.RData")

#Standard Logit setup
lhood <- function(par, X, y) {
  Xbeta <- X%*%par
  -sum(y*Xbeta - log(1 + exp(Xbeta)))
}

grad <- function(par, X, y) {
  p <-  1/(1 + exp(-X%*%par))
  -crossprod(X,(y-p))
}

#A version with a ridge penality

lhoodL2 <- function(par, X, y, prec) {
  Xbeta <- X%*%par
  -(sum(y*Xbeta - log(1 + exp(Xbeta))) -.5*sum(par^2*prec))
}

gradL2 <- function(par, X, y, prec) {
  p <-  1/(1 + exp(-X%*%par))
  -(crossprod(X,(y-p)) -par*prec)
}

lhoodL2(rep(0, ncol(X1)), X1, y, 1)

init <- rep(0, ncol(X1))
X <- Leukemia$x
y <- Leukemia$y
X1 <- cbind(1, X) #add an intercept

#Compare the Ridge version to optim
library(microbenchmark)
optim.out <- optim(init, lhoodL2, gradL2, method = "L-BFGS-B", X=X1, y=y, prec=2)
lbfgs.out <-lbfgs(lhoodL2, gradL2, init, invisible=1, X=X1, y=y, prec=2)
optim.out$value
lbfgs.out$value

#takes about 150 milliseconds

microbenchmark(optim.out <- optim(init, lhoodL2, gradL2, method = "L-BFGS-B", X=X1, y=y, prec=2),
               lbfgs.out <- lbfgs(lhoodL2, gradL2, init, invisible=1, X=X1, y=y, prec=2))

#Double check that its coded correctly
eval <- rnorm(ncol(X1))
library(numDeriv)
numeric <- numDeriv::grad(lhoodL2, x=eval, X=X, y=y, prec=1)
analytic <- gradL2(eval, X=X, y=y, prec=1)
all.equal(numeric,as.numeric(analytic))
#Hooray!

#############
#using the package with covariates
#############

#we can't pass it covariates right now
# but a hack solution is to just reference elements in the global environment

#we have to fix the precision in the global environment
prec <- 2

lhoodhack <- function(par) {
  Xbeta <- X1%*%par
  -(sum(y*Xbeta - log(1 + exp(Xbeta))) -.5*sum(par^2*prec))
}

gradhack <- function(par) {
  p <-  1/(1 + exp(-X1%*%par))
  -(crossprod(X1,(y-p)) -par*prec)
}

output <- lbfgs(lhoodhack, gradhack, rep(0, ncol(X1)), invisible=1)
optim.out <- optim(rep(0, ncol(X1)), lhoodL2, gradL2, method = "L-BFGS-B", X=X1, y=y, prec=2)
output2 <- lbfgs(lhoodL2, gradL2, rep(0, ncol(X1)), invisible=1, X=X1, y=y, prec=2)
output$value
optim.out$value
output2$value
#yay!  they are the same
all.equal(optim.out$par,output$par)
#yep- basically the same

#We can benchmark but this will be somewhat unfair to our package
library(microbenchmark)
microbenchmark(output <- lbfgs(lhoodhack, gradhack, rep(0, ncol(X1)), invisible=1),
               optim.out <- optim(rep(0, ncol(X1)), lhoodL2, gradL2, method = "L-BFGS-B", X=X1, y=y, prec=2),
               output2 <- lbfgs(lhoodL2, gradL2, rep(0, ncol(X1)), invisible=1, X=X1, y=y, prec=2),
               output3 <- lbfgs(lhoodL2, gradL2, rep(0, ncol(X1)), invisible=1, X=X1, y=y, prec=2, m=5))
#Note two things: (a) its a hack, (b) we have different parameters
# in particular m and the function tolerance are super important


#############
#Check the L1 part
#############

library(glmnet)
mod <- glmnet(X, y, family=c("binomial"))
coef(mod)[,50] #answer
mod$lambda[50] #regularization parameter

prec <- 0 #note have to set the precision to 0 when doing L1 because penalty is under orthantwise_c

system.time(owl.old <- lbfgsOptimize(lhoodL2, gradL2, rep(1, ncol(X1)), invisible=1, orthantwise_c=1, 
                    linesearch_algorithm="LBFGS_LINESEARCH_BACKTRACKING",
                    orthantwise_start = 1,
                    orthantwise_end = ncol(X1),
                    X=X1, y=y, prec=0))

system.time(owl.new <- lbfgs(lhoodL2, gradL2, rep(1, ncol(X1)), invisible=1, orthantwise_c=1, 
                         linesearch_algorithm="LBFGS_LINESEARCH_BACKTRACKING",
                         orthantwise_start = 1,
                         orthantwise_end = ncol(X1),
                         X=X1, y=y, prec=0))


which(coef(mod)[,70]!=0)
which(owl$par!=0)
#system("R CMD build RlibLBFGS")

mod$lambda[70]

######
# Poisson Regression
######

# Let's start with a basic poisson regression then we can update to more complex forms.
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

#just double checking the coding
eval <- rnorm(ncol(X))
library(numDeriv)
numeric <- numDeriv::grad(pois.ll, x=eval, X=X, y=y, prec=0)
analytic <- pois.grad(eval, X=X, y=y, prec=0)
all.equal(numeric,as.numeric(analytic))

optim.out <- optim(rep(0,5), pois.ll, pois.grad, X=X, y=y, method="L-BFGS-B")
cbind(as.numeric(coef(glm.D93)),optim.out$par)
#very minor differences but basically the same (its a convergence tolerance difference)


##Poisson example from glmnet
library(glmnet)
N=500; p=20
nzc=5
x=matrix(rnorm(N*p),N,p)
beta=rnorm(nzc)
f = x[,seq(nzc)]%*%beta
mu=exp(f)
y=rpois(N,mu)
fit=glmnet(x,y,family="poisson", standardize=FALSE)
C <- fit$lambda[25]*nrow(x)
coef(fit)[,25]
library(RlibLBFGS)
X <- x
#X <- t(t(x)/apply(x, 2, sd))
X1 <- cbind(1,X)

<<<<<<< HEAD
owl <- lbfgsOptimize(pois.ll,pois.grad, rep(0, ncol(X1)), X=X1, y=y, prec=0, 
                     invisible=1, orthantwise_c=C, 
                     linesearch_algorithm="LBFGS_LINESEARCH_BACKTRACKING",
                     orthantwise_start = 1,
                     orthantwise_end = ncol(X1))
=======

microbenchmark(
  owl.old <- lbfgsOptimize(pois.ll,pois.grad, rep(0, ncol(X1)), X=X1, y=y, prec=0, 
                       invisible=1, orthantwise_c=C, 
                       linesearch_algorithm="LBFGS_LINESEARCH_BACKTRACKING",
                       orthantwise_start = 1,
                       orthantwise_end = ncol(X1)),
  owl.new <- lbfgs(pois.ll,pois.grad, rep(0, ncol(X1)), X=X1, y=y, prec=0, 
                           invisible=1, orthantwise_c=C, 
                           linesearch_algorithm="LBFGS_LINESEARCH_BACKTRACKING",
                           orthantwise_start = 1,
                           orthantwise_end = ncol(X1))
)
>>>>>>> origin/cfunctions

cbind(owl$par,as.numeric(coef(fit)[,25]))

#okay let's do some quick timings
library(microbenchmark)
microbenchmark(glmnet(x,y,family="poisson", standardize=FALSE),
               owl <- lbfgs(pois.ll,pois.grad, rep(0, ncol(X1)), invisible=1, orthantwise_c=C, 
                                    linesearch_algorithm="LBFGS_LINESEARCH_BACKTRACKING",
                                    orthantwise_start = 1,
                                    orthantwise_end = ncol(X1),
                                    X=X1, y=y, prec=0))
# Yay its faster
# note this is a large N small p regime vs. leukemia which is small N and large p.
