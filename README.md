# Bayesian inference and prediction in linear regression models with continous and discrete response variables

Currently implemented in the 'bayes-linear' package are: 

1.) a binary probit model using variational inference and MCMC (Gibbs sampling) 
2.) a Student-t linear regression model for robust inference in case of heavy-tailed response variables using Gibbs sampling. For the MCMC probit model also Bayesian residuals as in [Albert and Chib (1995)](https://apps.olin.wustl.edu/faculty/chib/papers/albertchib95.pdf) are computed, which can be used for model diagnosis, outlier detection etc. 
3.) For active learning in binary classification problems the [Bayesian Active Learning by Disagreement (BALD) algorithm](https://arxiv.org/abs/1112.5745) has been implemented here for the Bayesian probit model.

All implementations are purely based on numpy, scipy and pandas.    
