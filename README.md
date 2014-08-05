lbfgs: Efficient L-BFGS Optimization in R
======

A wrapper built around the libLBFGS optimization library written by Naoaki Okazaki. The `lbfgs` package implements both the Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) and the Orthant-Wise Quasi-Newton Limited-Memory (OWL-QN) optimization algorithms. The L-BFGS algorithm solves the problem of minimizing an objective, given its gradient, by iteratively computing approximations of the inverse Hessian matrix. The OWL-QN algorithm finds the optimum of an objective plus the L1-norm of the problem's parameters, and can be used to train log-linear models with L1-regularization. The package offers a fast and memory-efficient implementation of these optimization routines, which is particularly suited for high-dimensional problems. The `lbfgs` package compares favorably with other optimization packages for R in microbenchmark tests. This readme is just to keep ideas for brainstorming.

Features for the Package
-----
* mirroring of optim syntax, possibly also defaults
* we may want to add in optional checks of the gradient using finite differences. (imported from another package)

To Do
-----
* When its ready to go we should email the CS people who use R and owl-qn (Noah Smith, Brendan etc.)


Vignette
-----
Outline
* Intro
	* two algorithms: LBFGS/OWL-QN focus on what they do well
	* pitch: drop-in replacement to optim.  Faster and with sparsity.
    * comparison to existing packages: penalized, glmnet etc. liblinear API interface is also relevant- likely to be faster but restricted to particular models.  
	* citations out to libLBFGS, Rcpp etc
* Algorithms
	* A brief self-contained introduction to the algorithms
	* LBFGS - Nocedal and Wright, first order, large set of parameters etc.
	* OWLQN - Andrew and Gao L1 penalty by restricted search etc. (maybe a very brief contrast to other L1 algs)
	* When to use the Package - embedded in iterative scheme, don't want to solve the full path, not a standard glm etc. 
* Implementation
	* How to use the API: underlying libraries Rcpp
	* Briefly what options are available, differences in optim vs. libLBFGS defaults
* Simple Benchmarks: Logistic Regression
	* Logistic regression benchmarks for the Leukemia data.  Idea is to focus on artificial but direct comparisons here.
	* Ridge Regression: optim vs. libLBFGS vs. glmnet (this is the LBFGS direct comparison)
	* L1 Logistic Regression: libLBFGS vs. glmnet vs. penalized (this is the OWL-QN comparison).  Give the direct replication of the glmnet table to draw the contrast to when each is useful.
* Case Study: Multinomial Logistic Regression
	* Set up the case study.  mlogit likelihood with L1 penalty- shows up in STM/SAGE/InverseRegression but also useful in its own right.  Many parameters, but also unlike other R packages we are in the regime of large outcome dimension.
	* Give the derivation.  Setup up the problem.
	* Benchmarks
	* Extension to the independent Poisson case re: Taddy
	* Benchmarks
	* Full STM example with timing comparisons (implicitly including how easy it is to use in package)
* Extensions
	* Passing C++ functions (if possible try to give a sense of the tradeoffs involved here)
	* Explain how to do other types of regularization schemes
	* Elastic net regularization (basically just include the ridge penalty and also specify L1)
	* Adaptive Lasso (I'm not even sure this is possible in libLBFGS but it would involve penalizing different components by different levels.  Presumably this can be done by selectively rescaling the predictors.  It is an odd hack though which is unlikely to be particularly efficient as the number of observations grows.)
* Conclusion
	* Draw the contrast between what is offered here and what is offered in other packages
	* Essentially the pitch is- it is very good at what it does, but there are lots of reasons an applied user would want to just use glmnet etc.





