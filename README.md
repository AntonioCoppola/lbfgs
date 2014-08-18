Efficient L-BFGS and OWL-QN Optimization in R
======

A wrapper built around the [libLBFGS](http://www.chokkan.org/software/liblbfgs/) optimization library written by Naoaki Okazaki. The `lbfgs` package implements both the Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) and the Orthant-Wise Quasi-Newton Limited-Memory (OWL-QN) optimization algorithms. The L-BFGS algorithm solves the problem of minimizing an objective, given its gradient, by iteratively computing approximations of the inverse Hessian matrix. The OWL-QN algorithm finds the optimum of an objective plus the L1-norm of the problem's parameters, and can be used to train log-linear models with L1-regularization. The package offers a fast and memory-efficient implementation of these optimization routines, which is particularly suited for high-dimensional problems. The `lbfgs` package compares favorably with other optimization packages for R in microbenchmark tests. A vignette is forthcoming.

Installation and Usage
-----
Download the package tarball and build using R commands, or alternatively instally directly from Github using Hadley Wickham's [devtools](https://github.com/hadley/devtools) package. The R command is:
```
library(devtools)
install_github("lbfgs", "AntonioCoppola")
```
For usage, please refer to the documentation and to the PDF manual.

