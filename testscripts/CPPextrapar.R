# Rosenbrock function in C++: Implementation with extra parameters
# Not yet functional. Will crash R if you run it.

objective2.include <- 'Rcpp::NumericVector rosenbrock(SEXP xs, SEXP env) {
  Rcpp::NumericVector x(xs);
  Rcpp::Environment e(env);
  
  std::vector<double> param = e["par"];
  Rprintf("%f\\n", param[0]);
  
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
  Rcpp::Environment e(env);
  
  std::vector<double> param = e["par"];
  Rprintf("%f\\n", param[1]);
  
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

output <- lbfgs(objective(), gradient(), c(-1.2,1), invisible=1)

output