import sys, time, theano
import pickle as pkl
import theano.tensor as T
import pymc3 as pm
import numpy as np
import pandas as pd
from warnings import filterwarnings
from pymc3.theanof import set_tt_rng, MRG_RandomStreams
from sklearn.base import BaseEstimator, ClassifierMixin

filterwarnings('ignore')


class bnn(BaseEstimator, ClassifierMixin):  
  
    """Bayesian 2-hidden layer FNN with informative prior"""

    def __init__(self, dec_thresh = 0.5, bnn_kwargs=None, sample_kwargs=None):
       self.dec_thresh = dec_thresh
       self.bnn_kwargs = bnn_kwargs 
       self.sample_kwargs = sample_kwargs

        
    def fit(self, X, y, ndraws=500, batch_size=100):
        
      self.xtrain = X  
      self.N, self.K = X.shape
  
      if self.bnn_kwargs is None:
         self.bnn_kwargs = {}

      if self.sample_kwargs is None:
         self.sample_kwargs = {'chains': 2, 'progressbar':True, 'tune': 500}         # use multiple chains to check for convergence
      #self.ann_input = pm.Minibatch(X, batch_size=batch_size)
      #self.ann_output = pm.Minibatch(y, batch_size=batch_size)
      
      self.ann_input = theano.shared(X)                    # shared variables so they keep their values between training iterations (updates)
      self.ann_output = theano.shared(y)

      self.model = self.bnn_func(self.ann_input, self.ann_output, **self.bnn_kwargs)

      with self.model:
        
          #self.start = pm.find_MAP()
          #self.inference = pm.NUTS()   
          #self.trace = pm.sample(5000, self.inference, start=self.start)
          
          # fit model via MCMC
          #self.trace = pm.sample(**self.sample_kwargs)
          #self.ppc_draws_train = pm.sample_posterior_predictive(trace = self.trace, samples=500, progressbar=True)
          
        # fit model via MCMC
        #----------------------
        #self.trace = pm.sample(**self.sample_kwargs)
        
        # fit model via variational inference:
        self.inference = pm.ADVI()                   # Note, that this is a mean-field approximation so we ignore correlations in the posterior.
        #self.inference = pm.SVGD(n_particles=500, jitter=1)
        
        tracker = pm.callbacks.Tracker(
               mean = self.inference.approx.mean.eval,  # callable that returns mean
               std = self.inference.approx.std.eval  # callable that returns std
        )
        
        self.approx = pm.fit(n=50000, method = self.inference, callbacks=[tracker])      # n: number of iterations
        self.trace = self.approx.sample(**self.sample_kwargs)    # sample from the variational distr.

        
        # sample posterior predictive
        ppc_train = pm.sample_posterior_predictive(self.trace, samples=ndraws, progressbar=True) 
        
        # Use probability of > 0.5 to assume prediction of class 1
        pred_train = ppc_train['out'].mean(axis=0) > 0.5
  
        # sample from prior predictive / marg. likelihood
        #self.marg_like = pm.sample_prior_predictive(samples=500) 
      return ppc_train, pred_train

    
    @staticmethod
    #def forward(x, *weights):
    def forward(x, w1, w1_2, w_out):
        act_1 = T.nnet.relu(pm.math.dot(x, w1))                      # intercept in design matrix -> bias is part of w1
        act_2 = T.nnet.relu(pm.math.dot(act_1, w1_2))
        #act_3 = T.nnet.relu(pm.math.dot(act_2, w2_3))
        y = pm.math.sigmoid(pm.math.dot(act_2, w_out))
        return y       
    
    def bnn_func(self, ann_input, ann_output, n_hidden1 = 5, n_hidden2 = 5, n_hidden3 = None):

        # Initialize random weights between each layer    
        init_1 = np.random.randn(self.K, n_hidden1).astype(float)
        init_2 = np.random.randn(n_hidden1, n_hidden2).astype(float)
        #init_3 = np.random.randn(n_hidden2, n_hidden3).astype(float)
        init_out = np.random.randn(n_hidden2).astype(float)

        with pm.Model() as model_def:

            # Priors:
            #---------
            #lam = 1 # 1e-10               # L1/L2 penalization hyperparameter
            
            lam = pm.Gamma('lambda',alpha = 2, beta=2, shape=1)
            
            # Weights from input to hidden layer
            #weights_in_1 = pm.Laplace('w_in_1', 0, b = 1/lam, 
            #                         shape=(self.K, n_hidden1), 
            #                         testval = init_1)

            # 2nd hidden layer: Weights from 1st to 2nd layer
            #weights_1_2 = pm.Laplace('w_1_2', 0, b = 1/lam, 
            #                        shape=(n_hidden1, n_hidden2), 
            #                        testval=init_2)
            
            # Weights from hidden layer to output
            #weights_2_out = pm.Laplace('w_2_out', 0, b = 1/lam, 
            #                          shape = (n_hidden2,), 
            #                          testval=init_out)

            # Weights from input to hidden layer  
            weights_in_1 = pm.Normal('w_in_1', 0, sd = 1/lam,            # precision lambda 
                                     shape=(self.K, n_hidden1), 
                                     testval=init_1)

            # 2nd hidden layer: Weights from 1st to 2nd layer
            weights_1_2 = pm.Normal('w_1_2', 0, sd=1/lam, 
                                    shape=(n_hidden1, n_hidden2), 
                                    testval=init_2)

            # 3nd hidden layer: 
            #weights_2_3 = pm.Normal('w_2_3', 0, sd=1/lam, 
            #                        shape=(n_hidden2, n_hidden3), 
            #                        testval=init_3)

            
            # Weights from hidden layer to output
            weights_out = pm.Normal('w_out', 0, sd=1/lam, 
                                      shape = (n_hidden2,), testval=init_out)

            # Build neural-network:
            #act_1 = T.nnet.relu(pm.math.dot(ann_input, weights_in_1))
            #act_2 = T.nnet.relu(pm.math.dot(act_1, weights_1_2))
            #act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_in_1))
            #act_2 = pm.math.tanh(pm.math.dot(act_1, weights_1_2))
            #act_out = pm.math.sigmoid(pm.math.dot(act_2, weights_2_out))

            act_out = bnn.forward(ann_input, weights_in_1, weights_1_2, weights_out)
            
            # Binary classification -> Bernoulli likelihood
            out = pm.Bernoulli('out', 
                               act_out,
                               observed = ann_output,    # observed rv! Given the data
                               total_size=self.N) 
        return model_def

    """    
    def predict(self, X=None, y=None, **sc):   
        
        try:
             getattr(self, "ppc_draws")
             labels_out = (self.ppc_draws['out'].mean(axis=0) > self.dec_thresh)*1   
             print("Using available ppc draws.") 
            
        except AttributeError as e:
             print(e)
             #raise RuntimeError("You must train classifer before predicting data!")
             print("No pp draws available -> draw from pp distr.")
             labels = self.score(X=X, **sc) 
             labels_out = (labels.ppc_draws['out'].mean(axis=0) > self.dec_thresh)*1
        finally:    
             return labels_out

          
    def score(self, X, y=None, ndraws=500):
      
        # Sample from posterior predictive distr.
        
        self.ann_input.set_value(X)      # Changing values here will also change values in the model
        #self.ann_output.set_value(y)
        
        #del self.ppc_draws
        #delattr(self, "ppc_draws")
        
        with self.model:
          # Sample from the variational distribution (using NUTS sampler by default):
          self.ppc_draws = pm.sample_posterior_predictive(trace = self.trace, samples = ndraws, progressbar=True)   
          
        print("Sampled from posterior predictive distr.")  
        return self  
    """      
      
    def score_new(self, X, ndraws=500):
     
        # create symbolic input
        x = T.matrix('X')

        # symbolic number of samples is supported, we build vectorized posterior on the fly
        n = T.iscalar('n')

        # Do not forget test_values or set theano.config.compute_test_value = 'off'
        x.tag.test_value = np.empty_like(self.ann_input)
        n.tag.test_value = 100

        _sample_proba = self.approx.sample_node(self.model.out.distribution.p,
                                           size = n,
                                           more_replacements = {self.ann_input: x})

        self.sample_proba = theano.function([x, n], _sample_proba) 
        return self.sample_proba(X, ndraws)
        
    
    def predict_new(self, X, **sc): 
      
        """Map scores to labels via decision threshold"""    
        
        labels_out = (self.score_new(X, ndraws=500).mean(0) > self.dec_thresh)*1   
        return labels_out
    
    