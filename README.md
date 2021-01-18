# Bayesian inference and prediction in linear regression models with continous and binary response variables

Currently implemented in the 'bayes-linear' package are 1.) a binary probit model using variational inference and MCMC (Gibbs sampling) and 2.) a Student-t linear regression model for robust inference in case of heavy-tailed response variables using Gibbs sampling. 

Also for active learning in binary classification problems the [Bayesian Active Learning by Disagreement (BALD) algorithm](https://arxiv.org/abs/1112.5745) has been implemented here for the Bayesian probit model.

All implementations are purely based on numpy, scipy and pandas.    
