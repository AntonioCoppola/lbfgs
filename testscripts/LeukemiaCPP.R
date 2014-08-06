# An example using L-BFGS to perform a Poisson regression using data from Golub, 
# Todd R., et al. "Molecular classification of cancer: class discovery and class 
# prediction by gene expression monitoring." Science 286.5439 (1999): 531-537.
# This time, the objective function and the gradient are implemented in C++.

library(lbfgs)
library(microbenchmark)
library(inline)
library(RcppArmadillo)

X <- Leukemia$x
y <- Leukemia$y
X1 <- cbind(1, X)

# Logit setup with ridge penality in R

lhoodL2 <- function(par, X, y, prec) {
  Xbeta <- X%*%par
  -(sum(y*Xbeta - log(1 + exp(Xbeta))) -.5*sum(par^2*prec))
}

gradL2 <- function(par, X, y, prec) {
  p <-  1/(1 + exp(-X%*%par))
  -(crossprod(X,(y-p)) -par*prec)
}

init <- rep(0, ncol(X1))
out.R <- lbfgs(lhoodL2, gradL2, init, X=X1, y=1, prec=1, invisible=1) 

out.R

# Logit setup with ridge penalty in C++

lhoodL2.inc <- 'Rcpp::NumericVector lhood(SEXP xs, SEXP env){
  arma::vec par = Rcpp::as<arma::vec>(xs);
  Rcpp::Environment e = Rcpp::as<Rcpp::Environment>(env);
  arma::mat X = Rcpp::as<arma::mat>(e["X"]);
  arma::vec y = Rcpp::as<arma::vec>(e["y"]);
  double prec = Rcpp::as<double>(e["prec"]);
  arma::mat Xbeta = X * par;
  double sum1 = sum(y % Xbeta - log(1 + exp(Xbeta)));
  arma::mat sum2 = sum(pow(par, 2 * prec));
  arma::vec out = -(sum1 - 0.5 * sum2);
  Rcpp::NumericVector ret = Rcpp::as<Rcpp::NumericVector>(wrap(out));
  return ret;
}
'

gradL2.inc <- 'Rcpp::NumericVector grad(SEXP xs, SEXP env){
  arma::vec par = Rcpp::as<arma::vec>(xs);
  Rcpp::Environment e = Rcpp::as<Rcpp::Environment>(env);
  arma::mat X = Rcpp::as<arma::mat>(e["X"]);
  arma::vec y = Rcpp::as<arma::vec>(e["y"]);
  double prec = Rcpp::as<double>(e["prec"]);
  arma::vec p = 1 / (1 + exp(-(X * par)));
  arma::vec grad = -((trans(X) * (y - p)) - par * prec);
  Rcpp::NumericVector ret = Rcpp::as<Rcpp::NumericVector>(wrap(grad));
  return ret;
}'

lhoodL2.body <- '
     typedef Rcpp::NumericVector (*funcPtr)(SEXP, SEXP);
     return(XPtr<funcPtr>(new funcPtr(&lhood)));
     '

gradL2.body <- '
     typedef Rcpp::NumericVector (*funcPtr)(SEXP, SEXP);
     return(XPtr<funcPtr>(new funcPtr(&grad)));
'

lhoodL2.CPP <- cxxfunction(signature(), body=lhoodL2.body, 
                          inc=lhoodL2.inc, plugin="RcppArmadillo")

gradL2.CPP <- cxxfunction(signature(), body=gradL2.body, 
                         inc=gradL2.inc, plugin="RcppArmadillo")

lhood.test <- cxxfunction(signature())

env <- new.env()
env[["X"]] <- X1
env[["y"]] <- y
env[["prec"]] <- 1

out.R <- lbfgs(lhoodL2, gradL2, init, invisible=1, X=X1, y=y, prec=1)
out.CPP <- lbfgs(lhoodL2.CPP(), gradL2.CPP(), init, environment=env, invisible=1)
out.optim <- optim(init, lhoodL2, gradL2, method = "L-BFGS-B", X=X1, y=y, prec=1)

all.equal(out.optim$value, out.R$value, out.CPP$value)

microbenchmark(out.optim <- optim(init, lhoodL2, gradL2, method = "L-BFGS-B", X=X1, y=y, prec=1),   
               out.R <- lbfgs(lhoodL2, gradL2, init, invisible=1, X=X1, y=y, prec=1),
               out.CPP <- lbfgs(lhoodL2.CPP(), gradL2.CPP(), init, environment=env, invisible=1))
