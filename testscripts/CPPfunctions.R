# LBFGS Optimization Package
# C++ External Pointers Usage Examples
# Antonio Coppola, Harvard University
# July 2014

library(inline)
library(lbfgs)
library(Rcpp)
library(microbenchmark)

# Rosenbrock function in C++: Basic implementation, no extra parameters

objective.include <- 'Rcpp::NumericVector rosenbrock(SEXP xs) { 
  Rcpp::NumericVector x(xs);
  double x1 = x[0];
  double x2 = x[1];
  double sum = 100 * (x2 - x1 * x1) * (x2 - x1 * x1)  + (1 - x1) * (1 - x1);
  Rcpp::NumericVector out(1);
  out[0] = sum;
  return(out);
}
'

gradient.include <- 'Rcpp::NumericVector rosengrad(SEXP xs) {
  Rcpp::NumericVector x(xs);
  double x1 = x[0];
  double x2 = x[1];
  double g1 = -400 * x1 * (x2 - x1 * x1) - 2 * (1 - x1);
  double g2 = 200 * (x2 - x1 * x1);
  Rcpp::NumericVector out(2);
  out[0] = g1;
  out[1] = g2;
  return(out);
}'

objective.body <- '
     typedef Rcpp::NumericVector (*funcPtr)(SEXP);
     return(XPtr<funcPtr>(new funcPtr(&rosenbrock)));
     '

gradient.body <- '
     typedef Rcpp::NumericVector (*funcPtr)(SEXP);
     return(XPtr<funcPtr>(new funcPtr(&rosengrad)));
'

objective <- cxxfunction(signature(), body=objective.body, 
                         inc=objective.include, plugin="Rcpp")

gradient <- cxxfunction(signature(), body=gradient.body, 
                         inc=gradient.include, plugin="Rcpp")

output <- lbfgs(objective(), gradient(), c(-1.2,1), invisible=1)

output

# Rosenbrock function in C++: Silly extra parameters demonstration

objective2.include <- 'Rcpp::NumericVector rosenbrock(SEXP xs, SEXP env) { 
  Rcpp::NumericVector x(xs);
  Rcpp::Environment e = Rcpp::as<Rcpp::Environment>(env);
  CharacterVector v = e["names"];
  Rcout << "Greetings from " << v[0] << " and " << v[1] << std::endl;
  double x1 = x[0];
  double x2 = x[1];
  double sum = 100 * (x2 - x1 * x1) * (x2 - x1 * x1)  + (1 - x1) * (1 - x1);
  Rcpp::NumericVector out(1);
  out[0] = sum;
  return(out);
}
'

gradient2.include <- 'Rcpp::NumericVector rosengrad(SEXP xs, SEXP env) {
  Rcpp::NumericVector x(xs);
  double x1 = x[0];
  double x2 = x[1];
  double g1 = -400 * x1 * (x2 - x1 * x1) - 2 * (1 - x1);
  double g2 = 200 * (x2 - x1 * x1);
  Rcpp::NumericVector out(2);
  out[0] = g1;
  out[1] = g2;
  return(out);
}'


objective2.body <- '
     typedef Rcpp::NumericVector (*funcPtr)(SEXP, SEXP);
     return(XPtr<funcPtr>(new funcPtr(&rosenbrock)));
     '

gradient2.body <- '
     typedef Rcpp::NumericVector (*funcPtr)(SEXP, SEXP);
     return(XPtr<funcPtr>(new funcPtr(&rosengrad)));
'

objective2 <- cxxfunction(signature(), body=objective2.body, 
                         inc=objective2.include, plugin="Rcpp")

gradient2 <- cxxfunction(signature(), body=gradient2.body, 
                        inc=gradient2.include, plugin="Rcpp")

env <- new.env()
env[["names"]] <- c("Antonio", "Brandon")

output <- lbfgs(objective2(), gradient2(), c(-1.2,1), 
                environment=env, invisible=1)
output

# Microbenchmark comparison: C++ vs. R objects vs. Optim

objective.R <- function(x) {   
  100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
}

gradient.R <- function(x) {
  c(-400 * x[1] * (x[2] - x[1]^2) - 2 * (1 - x[1]),
    200 * (x[2] - x[1]^2))
}

microbenchmark(out.CPP <- lbfgs(objective(), gradient(), c(-1.2,1), invisible=1),
               out.R <- lbfgs(objective.R, gradient.R, c(-1.2,1), invisible=1),
               out.optim <- optim(c(-1.2,1), objective.R, gradient.R, method="L-BFGS-B")) 

