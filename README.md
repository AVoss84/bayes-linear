# Bayesian inference for linear models with continous and discrete responses using MCMC and VI

Currently implemented in the 'bayes-linear' package are: 

1. Binary probit model using 
   1. Mean field variational inference (VI)
   1. Markov Chain Monte Carlo (MCMC) sampling (Gibbs sampling). Here also the posterior density function of the residuals is computed, which can be used for model diagnosis, outlier detection etc., see [Albert and Chib (1995)](https://apps.olin.wustl.edu/faculty/chib/papers/albertchib95.pdf) for details.
2. Student-t linear regression model for robust inference in case of heavy-tailed response variables using Gibbs sampling. 
3. For active learning in binary classification problems the [Bayesian Active Learning by Disagreement (BALD) algorithm](https://arxiv.org/abs/1112.5745), see also [Batch BALD](https://arxiv.org/abs/1906.08158), has been implemented here for the Bayesian probit model.

All implementations are purely based on numpy, scipy and pandas.    
