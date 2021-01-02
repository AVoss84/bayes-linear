
import numpy as np                # import numpy module
from numpy import random as r
import matplotlib.pyplot as plt
from matplotlib import rc               # for LaTex
from os import chdir
import pandas as pd
from scipy import sparse
from scipy.stats import t, gamma
from copy import deepcopy
from math import sqrt
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.linear_model import HuberRegressor


class bstudent:

    def __init__(self, nu = 5, verbose = True):                   # required arguments
        self.nu = nu   
        self.nu0 = 25
        self.verbose = verbose

    def log_digamma(self, x, shape, rate):
        return gamma.logpdf(1/x, a = shape, scale = 1/rate) - 2*np.log(x)  # last term is the Jacobian from z=1/x change of variables

    #@staticmethod
    def ols(self, X, Y):   
        """
        Computes the OLS estimates
        """ 
        self.X = deepcopy(X)
        self.Y = deepcopy(Y) 
        xx1 = np.linalg.inv(np.dot(X.T,X))    # (X'X)^-1
        xy = np.dot(X.T,Y)
        self.betas = np.dot(xx1,xy)
        self.yhat = np.dot(X,self.betas)
        self.res = Y - self.yhat            # set additional attributes
        self.r2 = 1 - (np.var(self.res)/np.var(self.Y))
        #return self

    def fit(self, X, Y, MCsim = 2000, burnin = 1000):
        """
        Gibbs sampler
        """        
        m = MCsim + burnin
        N, p = X.shape
        # Set prior hyperparameter:
        g = 100
        T0 = np.dot(X.T,X)/g   # prior precision matrix of regression coefficients; initialize with Zellner's g prior
        n0 = 0                # prior sample size
        s0 = 1                 # prior guess of error standard deviation
        b0 = np.full((p), 0.0, dtype = float)               # prior mean vector of regression coefficients
        beta = np.full((p,m), 0.0, dtype = float)
        sigma = np.full((m), 0.0, dtype = float)
        nus = np.full((m), 0.0, dtype = float)
        lambdas = np.full((N,m), 1.0, dtype = float)
        self.ols(X,y)                      # fit via OLS

        #beta[:,0] = lm.coef_
        nus[0] = self.nu
        beta[:,0] = self.betas
        sigma[0] = sqrt(sum(self.res**2)/(N-p))
        N1 = n0 + N
        mean_yhat = mean_eps = 0

        for i in range(1,m):        
            if (i%1000 == 0) & self.verbose: print(i)

            # Update regression coefficients
            #--------------------------------
            tau = (1/sigma[i-1]**2)                # precision
            Lam = np.diag(lambdas[:,i-1])               # prior mean vector of regression coefficients

            B1 = np.linalg.inv(T0 + tau*np.dot(np.dot(X.T,Lam),X))
            b1 = np.dot(B1, np.dot(T0, b0) + tau*np.dot(np.dot(X.T,Lam),y))
            beta[:,i] = r.multivariate_normal(mean = b1, cov = B1)

            # Update Latent precisions:
            #----------------------------
            yhat = np.dot(X, beta[:,i])       # linear predictions
            e = y - yhat
            #mean_eps = (mean_eps*(i-1) + e)/i  # residuals
            for j in range(N):
                lambdas[j,i] = r.gamma(shape = (self.nu + 1)/2, scale = 1/((self.nu + tau*e[j]**2)/2) )  # Note: scale = 1/rate . In R code rate is used instead of sclae

            # Update error standard deviation:
            #----------------------------------
            shape = N1/2
            Lam_i = np.diag(lambdas[:,i])
            rate = (n0 * s0**2 + np.dot(np.dot(e.T, Lam_i), e))/2
            sigma[i] = sqrt(1/r.gamma(shape, scale = 1/rate))
 
            # Klappt noch nicht, zero accept! 
            nu1 = r.normal(loc = self.nu, scale = 1, size=1)[0]
            if nu1 > 0:
                num = np.sum(bs.log_digamma(lambdas[:,i], nu1/2, nu1/2) + gamma.logpdf(nu1, a=1, scale=1/self.nu0))
                den = np.sum(bs.log_digamma(lambdas[:,i], self.nu/2, self.nu/2) + gamma.logpdf(self.nu, a=1, scale=1/self.nu0))
                lacc = num-den
                if np.log(r.uniform(size=1)[0]) < lacc :
                    self.nu = nu1
                    nus[i] = self.nu

        # Discard burnin draws:
        #-----------------------
        if self.verbose: print("Discarding burnin draws.")
        betas = np.delete(beta,np.s_[:burnin], axis = 1)
        nus = np.delete(nus,np.s_[:burnin], axis = 0)
        sigmas = np.delete(sigma,np.s_[:burnin], axis = 0)
        lambdass = np.delete(lambdas,np.s_[:burnin], axis = 1)
        return betas, sigmas, lambdass, nus


bs = bstudent()
#bs.ols(X,y)
betas, sigmas, lambdass, nus = bs.fit(X,y) 


#-------
# DGP:
#-------
N = 200
k = 2
dof = 50
sigma_true = .05
#np.random.seed(123)            # set seed
X = np.column_stack((np.ones(N),r.rand(N,k)))
#eps = r.normal(loc = 0.0, scale = sigma_true, size = N)    # normal
eps = sigma_true * r.standard_t(df = dof, size = N)               # Student-t
beta_true = np.array((0.2,1.23,0.34))                     # true betas
y = np.dot(X, beta_true) + eps                             # Signal
#-----------------------------
ols(X,y).r2


# Plot density of response:
#----------------------------
count, bins, ignored = plt.hist(y, 30, density = True)
plt.xlabel('y')
plt.title('Distribution of y')
plt.ylabel('density')
plt.show()


#s = pd.Series(sigmas)
#plt.figure()
#autocorrelation_plot(s)
#s.plot()


# Bayes estimate:
#------------------
#post_med = np.median(betas, axis = 1)
post_beta = np.mean(betas, axis = 1)
post_sig = np.mean(sigmas)
post_lambda = np.mean(lambdass, axis = 1)
#------------------------------------------
print("Bayes estimate betas:", post_beta)
#print("\nOLS estimates:",fit.betas)
print("\nHuber estimates betas:",huber.coef_)
print("\nTrue coefficients betas:", beta_true)

print("Bayes estimate sigma:", post_sig)
#print("\nHuber estimates sigma:",huber.scale_)
print("\nTrue coefficient sigma:", sigma_true)


# Posterior estimate:
#---------------------------
R2(y-np.dot(X, post_beta), y)
fit.r2              # classical R2

#------------------------------------------------------------
ig, ax = plt.subplots(1, 1)
df = 5
x = np.linspace(t.ppf(0.01, df), t.ppf(0.99, df), 100)
ax.plot(x, t.pdf(x, df), 'r-', lw=5, alpha=0.6, label='t pdf')
ax.title(r"Student-t density pdf")
plt.show()
#---------------------------------------------------------------


# Plot beta coeff. MCMC draws:
#=====================================================================================
for j in range(0,p):
 lw = 1
 #plt.subplot(2, 1, j)
 plt.figure(figsize=(6, 5))
 plt.plot(betas[j,:], color='C'+str(j), linewidth=lw,label="Beta " + str(j+1))
 plt.plot(post_beta[j], color='black', linewidth=lw, label=r"True value")
 plt.xlabel("Draws")
 plt.ylabel(r"value")
 plt.title(r"Trace plot of $\beta$ posterior draws")
 plt.legend(loc="best", prop=dict(size=12))
 plt.show()
 #------------------------------------------

 # Plot density of betas:
 #--------------------------
for j in range(0,p):
 count, bins, ignored = plt.hist(betas[j,:], 50, density = True, color='C'+str(j))
 plt.xlabel("beta" + str(j+1))
 plt.title('Posterior of beta'+ str(j))
 plt.ylabel('density')
 plt.show()
#=====================================================================================

# Sigma:
 #-------
 plt.figure(figsize=(6, 5))
 plt.plot(sigmas, color="black", linewidth=lw,label=r"$\sigma$")
 #plt.plot(post_med[j], color='black', linewidth=lw, label=r"True value")
 plt.xlabel(r"Draws")
 plt.ylabel(r"value")
 plt.title(r"Trace plot of $\sigma$ posterior draws")
 
#-----------------------------------------------------------------------------------
# Posterior Density estimate:

 count, bins, ignored = plt.hist(sigmas, 40, density = True, color='black')
 plt.xlabel(r"$\sigma$")
 plt.ylabel(r"value")
 plt.title(r"Trace plot of $\sigma$ posterior draws")
 plt.show()


# #############################################################################
# Fit the Bayesian Ridge Regression and an OLS for comparison
clf = BayesianRidge(n_iter=500, compute_score=True)

clf.fit(X, y)
beta_hat = clf.coef_

X[:10,:p]

n,k = X.shape
clf.get_params()
pred = clf.predict([[1,0.56,0.88]], return_std = True) ; pred      # yhat
yhat = pred[0]
se_yhat = pred[1]
R2 = clf.score(X,y) ; R2

ols = LinearRegression()
ols.fit(X, y)



