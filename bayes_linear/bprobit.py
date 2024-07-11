from copy import deepcopy
from tqdm import tqdm
from math import pi, lgamma
from scipy.special import digamma
from scipy.stats import norm
from numpy.linalg import det
import numpy as np
from numpy import linalg
from numpy.linalg import inv
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve


class BayesProbit(BaseEstimator):
    """
    Variational + MCMC Bayesian Binary Probit model inference
    """

    def __init__(self, seed: int = None, verbose: bool = True):
        self.seed = seed
        self.verbose = verbose

    # def __del__(self):
    #    print("Destructor called")

    def simulate_DGP(self, N, D, mu_theta=0, set_seed=None):
        """
        Draw sample from binary probit data generating process
        """
        if set_seed:
            np.random.seed(set_seed)
        X = np.zeros((N, D + 1))
        X[:, 0] = 1  # constant
        X[:, 1:] = np.random.random(
            (N, D)
        )  # * 2 - 1     # draw d-dim. uniform [-1, +1]
        # X[:,1:] = np.random.multivariate_normal(mean = np.zeros(D), cov = 1.1*np.eye(D), size = N)

        # True values of regression coeff. theta
        true_theta = np.random.normal(mu_theta, 1, D + 1)
        # true_theta = np.random.random(D+1) * 2 - 1     # draw d+1-dim uniform [-1, +1]

        # Obtain the vector with probabilities of success p using the probit link
        p = self.pnorm(np.dot(X, true_theta))

        # Generate binary observation data y
        y = np.random.binomial(1, p, N)
        return y, X, true_theta

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self

    def score(self, X, y):
        return self

    def qnorm(self, p, mu=0, sd=1):
        return norm.ppf(p, loc=mu, scale=sd)

    def pnorm(self, x, mu=0, sd=1):
        return norm.cdf(x, loc=mu, scale=sd)

    def entropy(self, p0, p1):
        return -(p0 * np.log2(p0) + p1 * np.log2(p1))

    def rtruncnorm(self, n, a=-np.inf, b=np.inf, mu=0, sigma=1):
        """
        Draw from (double) truncated normal distribution
        Inputs:
        n : size, humber of random draws
        a, b : left, right truncation points
        mu, sigma : mean and standard deviation of normal distr.

        Output:
        Array with draws of length n
        """
        assert b > a, "Upper truncation point must be strictly greater than lower!"
        u = np.random.uniform(size=n)
        x = self.qnorm(
            (1 - u) * self.pnorm((a - mu) / sigma) + u * self.pnorm((b - mu) / sigma)
        )  # see Lynch for example
        return mu + sigma * x


class BayesProbit_MCMC(BayesProbit):
    """
    Bayesian Binary Probit regression using Gibbs sampling ('Data augmentation')
    """

    def __init__(
        self,
        N_sim: int = 10000,
        burn_in: int = 5000,
        pred_mode: list = ["plug_in", "full"],
        thin: int = None,
        train_val_split: float = 0.3,
        random_state: int = None,
        prior_variance_beta: float = 1,
        seed: int = None,
        verbose: bool = True,
    ):

        self.N_sim = N_sim
        self.burn_in = burn_in
        self.seed = seed
        self.verbose = verbose
        self.thin = thin
        self.prior_variance_beta = prior_variance_beta
        self.pred_mode = pred_mode[0]
        self.test_size = train_val_split
        self.random_state = random_state
        super().__init__(self.seed, self.verbose)
        assert self.N_sim > self.burn_in, "Set N_sim >> burn_in !!"

    def fit(self, X, y):

        # Split data set into train/dev set for later optimal decision threshold determination:
        X, self.X_dev, y, self.y_dev = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        N, D = X.shape
        if self.seed is not None:
            np.random.seed(self.seed)

        # Conjugate prior on the coefficients \theta ~ N(theta_0, Q_0)
        theta_0 = np.zeros(D)
        Q_0 = np.diag([self.prior_variance_beta] * D)

        # Initialize parameters
        theta = np.zeros(D)
        z = np.zeros(N)
        self.theta_chain = np.zeros((self.N_sim, D))
        N1 = np.sum(y)  # Number of successes
        N0 = N - N1  # Number of failures
        mu_z = np.dot(X, theta)

        # Compute posterior variance of theta
        prec_0 = inv(Q_0)
        V = inv(prec_0 + np.dot(X.T, X))

        for t in tqdm(range(1, self.N_sim)):
            # Update Mean of z
            mu_z = np.dot(X, theta.T)

            # Draw latent variable z from its full conditional: z | \theta, y, X
            z[y == 0] = self.rtruncnorm(
                N0, a=-np.inf, b=0, mu=mu_z[y == 0].flatten(), sigma=1
            )
            z[y == 1] = self.rtruncnorm(
                N1, a=0, b=np.inf, mu=mu_z[y == 1].flatten(), sigma=1
            )

            # Compute posterior mean of theta
            M = np.dot(V, np.dot(prec_0, theta_0) + np.dot(X.T, z))

            theta = np.random.multivariate_normal(mean=M, cov=V, size=1)
            self.theta_chain[t, :] = theta

        # if self.verbose : print('Training finished!')

        if self.thin is not None:
            if self.verbose:
                print("Apply {}-thinning.".format(self.thin))
            index_thinning = np.array(range(self.theta_chain.shape[0]))
            self.theta_chain = self.theta_chain[
                index_thinning[index_thinning[0] :: self.thin], :
            ]  # take every thin'th value

        self.theta_final = self.theta_chain[self.burn_in :, :]
        if self.verbose:
            print("Discarding first {} draws.".format(self.burn_in))

        # Get posterior mean of \theta
        self.post_theta_est = np.mean(self.theta_final, axis=0)
        # self.post_theta_est = np.median(self.theta_final, axis=0)

        # Determine decision threshold based on hold-out set:
        # ------------------------------------------------------
        verbose = deepcopy(self.verbose)
        self.verbose = False
        # calculate roc curves on a hold-out dev set
        fpr, tpr, thresholds = roc_curve(self.y_dev, self.predict_proba(self.X_dev))
        # calculate geometrix mean of both rates for each threshold
        gmeans = np.sqrt(tpr * (1 - fpr))
        # locate the index of the largest geom. mean
        ix = np.argmax(gmeans)
        self.dec_thresh = thresholds[ix]
        self.verbose = verbose
        # ----------------------------------------------------------------------
        return self.post_theta_est  # 'beta hats'

    def score(self, X, y):
        """
        Calculate accuracy score.
        y: true labels associated with X
        """
        yhat = self.predict(X=X, y=y)
        self.accuracy = np.round(np.mean(y == yhat), 3)
        # if self.verbose : print('Accuracy : {}'.format(self.accuracy))
        return self.accuracy

    def predict_proba(self, X, y=None):
        """
        Calculate posterior class probabilities and residual posterior distribution for residual analysis, e.g. outlier detection (see Albert and Chib, 1994)
        """
        N = X.shape[0]
        if self.verbose:
            print("Using predictive mode: {}".format(self.pred_mode))
        self.p_pred = np.zeros(
            (self.N_sim - self.burn_in, N)
        )  # success posterior predictive prob.
        if y is not None:
            self.residuals = np.zeros((self.N_sim - self.burn_in, N))

        # Evaluate posterior predictive p.m.f.:
        # ----------------------------------------
        if self.pred_mode == "full":
            for j in tqdm(range(self.theta_final.shape[0])):
                self.p_pred[j, :] = self.pnorm(
                    np.dot(X, self.theta_final[j, :].T)
                ).flatten()  # given X (training data!)
                if y is not None:
                    self.residuals[j, :] = (
                        y - self.p_pred[j, :]
                    )  # Residual draws: epsilon(y, beta^{j}) given y observed, see page 4 ('first approach')

            # Draw from pred. distr.:
            # -------------------------
            # y_pred[t] = np.random.binomial(1, p_pred[:,t], N)
            pred_prob_estimate = np.mean(
                self.p_pred, axis=0
            )  # fully bayesian using the MCMC draws from the posterior; E(p_i | y) in Albert/Chib, 1995
        else:
            pred_prob_estimate = self.pnorm(
                np.dot(X, self.post_theta_est)
            )  # plug-in estimate (see Robert/Marin)

        if y is not None:
            self.resid_post_mean = self.residuals.mean(axis=0)
        return pred_prob_estimate

    def predict(self, X, y=None):
        """Predict labels"""
        return (self.predict_proba(X, y) > self.dec_thresh) * 1.0


class BayesProbit_VI(BayesProbit):
    """
    Variational Bayes Binary Probit model using coordinate ascent
    """

    def __init__(
        self,
        basis=None,
        seed: int = None,
        alpha_0=1e-1,
        beta_0=1e-1,
        max_iter=500,
        epsilon_conv=1e-5,
        train_val_split: float = 0.3,
        random_state: int = None,
        verbose: bool = True,
    ):

        self.seed = seed
        self.verbose = verbose
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.max_iter = max_iter
        self.epsilon_conv = epsilon_conv
        self.test_size = train_val_split
        self.random_state = random_state
        super().__init__(self.seed, self.verbose)

    def score(self, X, y):
        return np.round(np.mean(self.predict(X) == y), 3)

    def fit(self, X, y):
        """
        Train binary probit model via Coordinate ascent mean-field variational inference (CAVI)
        Input:
        alpha_0, beta_0 : shape hyperparameters of Gamma prior of precision of conjugate zero-mean normal prior of regression coeff. w
        basis: matrix \Omega(x) with basis functions applied to each column of design matrix X
        """
        # Split data set into train/dev set for later optimal decision threshold determination:
        X, self.X_dev, y, self.y_dev = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        N, D = X.shape
        L = np.full(self.max_iter, 0.0)  # Store the lower bounds
        E_z = np.zeros(N)

        XX = np.dot(X.T, X)  # precalculate inner product
        alpha = self.alpha_0 + 0.5 * D  # 'shape' of Gamma(\tau | \alpha, \beta)
        beta = self.beta_0  # rate' of Gamma(\tau | \alpha, \beta)
        S = np.linalg.inv(np.eye(D) + XX)  # covariance of q(w | m, S)
        m = np.zeros((D, 1))  # mean of q(w | m, S)
        L[0] = -1e40  # initialize lower bound (will be maximized!)
        robustify = 1e-10  # add small number to guarantee finite log values

        # Iterate to find optimal parameters
        for i in range(1, self.max_iter):
            # Update mean of q(z)
            mu = np.dot(X, m)
            mu_1 = mu[y == 1]  # Keep data where y == 1
            mu_0 = mu[y == 0]  # Keep data where y == 0

            # Compute expectation E[z]
            E_z[y == 1] = (mu_1 + norm.pdf(-mu_1) / (1 - self.pnorm(-mu_1))).squeeze()
            E_z[y == 0] = (mu_0 - norm.pdf(-mu_0) / self.pnorm(-mu_0)).squeeze()

            # Update parameters of q(w)
            S = np.linalg.inv(
                (alpha / beta) * np.eye(D) + XX
            )  # posterior covariance of q(w)
            m = np.dot(S, np.dot(X.T, E_z)).reshape(D, 1)  # posterior mean of q(w)
            beta = self.beta_0 + 0.5 * (np.dot(m.T, m) + np.trace(S))

            # -----------------------
            # Compute lower bound L:
            # -----------------------
            lb_p_zw_qw = (
                -0.5 * np.trace(np.dot(XX, np.dot(m, m.T) + S))
                + 0.5 * np.dot(mu.T, mu)
                + np.sum(
                    y * np.log(1 - self.pnorm(-mu) + robustify).squeeze()
                    + (1 - y) * np.log(self.pnorm(-mu) + robustify).squeeze()
                )
            )

            lb_pw = (
                -0.5 * D * np.log(2 * pi)
                + 0.5 * D * (digamma(alpha) - np.log(beta))
                - alpha / (2 * beta) * (np.dot(m.T, m) + np.trace(S))
            )

            lb_qw = -0.5 * np.log(max(det(S), 0.0001)) - 0.5 * D * (1 + np.log(2 * pi))

            lb_pa = (
                self.alpha_0 * np.log(self.beta_0)
                + (self.alpha_0 - 1) * (digamma(alpha) - np.log(beta))
                - self.beta_0 * (alpha / beta)
                - lgamma(self.alpha_0)
            )

            lb_qa = -lgamma(alpha) + (alpha - 1) * digamma(alpha) + np.log(beta) - alpha

            L[i] = lb_p_zw_qw + lb_pw + lb_pa - lb_qw - lb_qa  # lower bound

            # Show VB difference
            if self.verbose & (i % 10 == 0):
                print(
                    "Iter. {} Lower Bound {} - Delta LB {}".format(
                        i, round(L[i], 4), round(L[i] - L[i - 1], 4)
                    )
                )
            # Check if lower bound decreases
            if L[i] < L[i - 1]:
                print("Lower bound decreases!\n")
            # Check for convergence
            if L[i] - L[i - 1] < self.epsilon_conv:
                print("Converged!\n")
                break
            # Check if VB converged in the given maximum iterations
            if i == (self.max_iter - 1):
                print("Algorithm did not converge!\n")

        # Output posterior parameter of variational approx. and lower bound values on marg. likelihood
        self.model = dict(m=m, S=S, alpha=alpha, beta=beta, LowerB=L[2:i])

        # Determine empirical decision threshold based on hold-out set:
        # ---------------------------------------------------------------
        verbose = deepcopy(self.verbose)
        self.verbose = False
        # calculate roc curves on a hold-out dev set:
        fpr, tpr, thresholds = roc_curve(self.y_dev, self.predict_proba(self.X_dev))
        # calculate geometrix mean of both rates for each threshold
        gmeans = np.sqrt(tpr * (1 - fpr))
        # locate the index of the largest geom. mean
        ix = np.argmax(gmeans)
        self.dec_thresh = thresholds[ix]
        self.verbose = verbose
        # -------------------------------------------------------------------------------
        return self.model

    def predict_proba(self, X: np.array):
        """
        Compute variational predictive distribution of the probit model
        """
        # Predictive mean
        self.mu_pred = (X @ self.model["m"]).squeeze()
        # Predictive variance
        self.s_pred = np.sqrt(1 + np.diag(X @ self.model["S"] @ X.T))
        # Bernoulli variational posterior predictive mass function:
        Phi = self.pnorm(x=np.divide(self.mu_pred, self.s_pred))  # prob. of success
        return Phi

    def predict(self, X: np.array):
        """Predict labels"""
        return (self.predict_proba(X) > self.dec_thresh) * 1.0


class design_matrix(TransformerMixin):
    """
    Create design matrix based on a given basis function
    """

    def __init__(
        self, n_features, basis=["poly", "gauss"], add_bias=True, **basis_para
    ):
        self.basis = basis[0]
        self.n_features = n_features
        self.add_bias = add_bias

        if self.basis == "poly":
            self.meth = PolynomialFeatures(
                degree=self.n_features, include_bias=False, **basis_para
            )  # polynomial basis
        if self.basis == "gauss":
            self.meth = GaussianFeatures(
                N=self.n_features, **basis_para
            )  # Gaussian basis function

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for i in range(X.shape[1]):
            self.meth.fit(X[:, i].reshape(-1, 1))
            XX = self.meth.transform(X[:, i].reshape(-1, 1))
            if i == 0:
                A = XX
            else:
                A = np.concatenate((A, XX), axis=1)
        if self.add_bias:
            A = np.insert(A, 0, values=1, axis=1)  # add 1 at colum 0
        return A

    def fit_transform(self, X, y=None):
        return self.transform(X)


class GaussianFeatures(TransformerMixin):
    """Uniformly spaced Gaussian features for one-dimensional input"""

    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor

    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg**2, axis))

    def fit(self, X, y=None):
        # create N centers spread along the data range
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self

    def transform(self, X):
        return self._gauss_basis(
            X[:, :, np.newaxis], self.centers_, self.width_, axis=1
        )


def pnorm(x, mu=0, sd=1):
    return norm.cdf(x, loc=mu, scale=sd)


def entropy(p0, p1):
    return -(p0 * np.log2(p0) + p1 * np.log2(p1))


def BALD(X_pool, theta_train, b=10):
    """
    Bayesian Active Learning by Disagreement (BALD) for binary classif. case, i.e. for n = 1 and c = 2
    ------
    Input:
    ------
    X_pool : pandas dataframe of unlabelled examples
    theta_train: Random sample from posterior distribution of theta regression coefficients
    b : acquisition size, n = 1....b, see Batch BALD paper
    """
    thetas = deepcopy(theta_train)
    n_pool = X_pool.shape[0]  # number of unlabelled obs. x_pool
    c = 2  # number of classes
    k = thetas.shape[0]  # number of used MC samples
    # H_y_cond = np.zeros(b)
    argsmax = np.empty(b, int)
    maxi = np.empty(b, np.float32)
    Phis = np.zeros((k, n_pool))  # success posterior predictive prob.
    Pn = np.zeros((c, k))
    Pn1 = np.empty((0, k), np.float32)  # NULL array
    for j in range(k):
        Phis[j, :] = pnorm(
            np.dot(X_pool, thetas[j, :].T)
        ).flatten()  # success probab. Pr(y = 2|theta)

    A_n = ()
    for n in range(b):

        print("{} out of {}".format(n + 1, b))
        a_batch_bald = []
        for i in range(n_pool):  # over x in D_pool
            Phi_i = Phis[:, i]  # prob. of success in binary classif.

            Pn[0, :] = 1 - Phi_i
            Pn[c - 1, :] = Phi_i

            h = entropy(p0=1 - Phi_i, p1=Phi_i)
            E_H_cond = np.mean(
                h
            )  # Monte Carlo mean over posterior draws -> eqn. (9); marginal uncertainty / confidence

            H_y = entropy(p0=np.mean(1 - Phi_i), p1=np.mean(Phi_i))  # eqn. (12)
            # E_H_cond = np.sum(H_y_cond)     # sum over n=1...b
            ac = (
                H_y - E_H_cond
            )  # Disagreement between model parameters and unlabelled prediction; the larger - the more worth it is to actively labelling/learning them
            a_batch_bald.append(ac)

        # if n == 0: a_batch_bald0 = a_batch_bald

        # Find maximum disagreement between prediction y_hat and model parameters:
        index = np.argmax(np.array(a_batch_bald))
        argsmax[n] = index + n
        maxi[n] = np.array(a_batch_bald)[index]

        # -----------
        # Updates:
        # -----------
        A_n += (
            n,
            X_pool[index, :],
        )  # make union A_n-1 and x; save most promising examples for active learning
        Phis = np.delete(Phis, (index), axis=1)  # leave out x_current in next round
        n_pool -= 1

        # write final x in Pn
        Pn[0, :] = 1 - Phis[:, index]
        Pn[c - 1, :] = Phis[:, index]

        # and save Pn in Pn1 ('union')
        Pn1 = np.append(Pn1, Pn, axis=0)
    return A_n, argsmax, maxi  # , np.array(a_batch_bald0)
