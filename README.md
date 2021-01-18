# Bayesian inference and prediction in linear regression models with continous and discrete response variables

Currently implemented in the 'bayes-linear' package are: 

1. Binary probit model using 
   1. Mean field variational inference
   1. Markov Chain Monte Carlo sampling (Gibbs sampling). Here also the posterior density function of the residuals is computed, which can be used for model diagnosis, outlier detection etc., see [Albert and Chib (1995)](https://apps.olin.wustl.edu/faculty/chib/papers/albertchib95.pdf) for details.
2. Student-t linear regression model for robust inference in case of heavy-tailed response variables using Gibbs sampling. 
3. For active learning in binary classification problems the [Bayesian Active Learning by Disagreement (BALD) algorithm](https://arxiv.org/abs/1112.5745) has been implemented here for the Bayesian probit model.

All implementations are purely based on numpy, scipy and pandas.    
