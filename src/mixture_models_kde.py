#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:10:19 2024

@author: veroneze
"""

import numpy as np
from scipy.stats import rv_continuous, norm, weibull_min, gaussian_kde
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture


class GMMDist(rv_continuous):
    """
    Gaussian Mixture Model Distribution (GMM) class based on scipy.stats.rv_continuous
    """
    def __init__(self, n_components=2):
        """
        Initialize the GMM distribution.

        Parameters:
        - n_components: int, the number of Gaussian components in the mixture
        """
        super().__init__()
        self.name = 'GMM'
        self.n_components = n_components
        self.param_names = ['weights', 'means', 'stds']

    def fit(self, x, **kwds):
        """
        Fit the GMM to the data using a simple Expectation-Maximization (EM) algorithm.
        
        Parameters:
        - x: 1D array of data points to fit the GMM
        
        Returns:
        - (weights, means, stds): Tuple of fitted parameters
        """
        
        
        """
        FOR AN EM WITH MORE OPTIONS AND FLEXIBILITY,
        I DECIDED TO USE THE GaussianMixture FROM sklearn
        """
        # weights, means, stds = self._EM(x, **kwds)
        gmm = GaussianMixture(n_components=self.n_components, **kwds)
        try:
            gmm.fit(np.array(x).reshape(-1, 1))
        except Exception as e:
            raise e
        weights = gmm.weights_
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_).flatten()
        return weights, means, stds


    def _EM(self, x, **kwds):
        """
        Fit the GMM to the data using Expectation-Maximization (EM) algorithm.
        
        Parameters:
        - x: 1D array of data points to fit the GMM
        
        Returns:
        - (weights, means, stds): Tuple of fitted parameters
        """
        
        # Parameters for EM's stopping criteria
        max_iter = kwds['max_iter'] if 'max_iter' in kwds else 1000
        tol = kwds['tol'] if 'tol' in kwds else 0.001
        
        # Initialize means, stds, and weights
        weights = np.full(self.n_components, 1.0 / self.n_components)
        means = np.random.choice(x, self.n_components)
        stds = np.full(self.n_components, np.std(x))
        
        # Compute the log_likelihood for testing convergence
        prev_log_likelihood = np.sum(self.logpdf(x, weights, means, stds))
        
        # EM loop to fit the GMM
        for iteration in range(max_iter):
            print(iteration)
            # E-step: Calculate responsibilities
            responsibilities = np.zeros((len(x), self.n_components))
            for k in range(self.n_components):
                responsibilities[:, k] = weights[k] * norm.pdf(x, means[k], stds[k])
            responsibilities /= responsibilities.sum(axis=1, keepdims=True)
            
            # M-step: Update weights, means, and stds
            for k in range(self.n_components):
                resp_sum = responsibilities[:, k].sum()
                weights[k] = resp_sum / len(x)
                means[k] = np.sum(responsibilities[:, k] * x) / resp_sum
                variance = np.sum(responsibilities[:, k] * (x - means[k])**2) / resp_sum
                stds[k] = np.sqrt(variance)
            
            # Check convergence
            log_likelihood = np.sum(self.logpdf(x, weights, means, stds))
            if abs(prev_log_likelihood - log_likelihood) < tol:
                break
            prev_log_likelihood = log_likelihood
        
        return weights, means, stds
    
    
    def pdf(self, x, weights, means, stds):
        """
        Probability Density Function (PDF) of the GMM.
    
        Parameters:
        - x: array-like, input data
        - weights: array-like, mixture weights
        - means: array-like, component means
        - stds: array-like, component standard deviations
    
        Returns:
        - pdf_values: array-like, PDF evaluated at x
        """
        
        pdf_values = np.zeros_like(x, dtype=float)
        for k in range(self.n_components):
            pdf_values += weights[k] * norm.pdf(x, means[k], stds[k])
        return pdf_values
    
    
    def logpdf(self, x, weights, means, stds):
        """
        Log of the Probability Density Function (log-PDF) of the GMM.

        Parameters:
        - x: array-like, input data
        - weights: array-like, mixture weights
        - means: array-like, component means
        - stds: array-like, component standard deviations

        Returns:
        - log_pdf_values: array-like, log-PDF evaluated at x
        """
        # Get the PDF values
        pdf_values = self.pdf(x, weights, means, stds)
        
        # Take the logarithm of the PDF values, handle zero values to avoid log(0)
        log_pdf_values = np.log(np.maximum(pdf_values, 1e-300))  # Small epsilon to avoid log(0)
        
        return log_pdf_values


    def cdf(self, x, weights, means, stds):
        """
        Cumulative Distribution Function (CDF) of the GMM.

        Parameters:
        - x: array-like, input data
        - weights: array-like, mixture weights
        - means: array-like, component means
        - stds: array-like, component standard deviations
        
        Returns:
        - cdf_values: array-like, CDF evaluated at x
        """
        
        cdf_values = np.zeros_like(x, dtype=float)
        for k in range(self.n_components):
            cdf_values += weights[k] * norm.cdf(x, means[k], stds[k])
        return cdf_values

    def rvs(self, weights, means, stds, size=1, random_state=None):
        """
        Generate random samples from the GMM.

        Parameters:
        - weights: array-like, mixture weights
        - means: array-like, component means
        - stds: array-like, component standard deviations
        - size: int, number of samples to generate
        - random_state: int, seed for random number generator

        Returns:
        - samples: array-like, generated random samples
        """
        
        rng = np.random.default_rng(random_state)
        samples = np.zeros(size)
        component_samples = rng.choice(self.n_components, size=size, p=weights)
        for k in range(self.n_components):
            mask = component_samples == k
            samples[mask] = norm.rvs(loc=means[k], scale=stds[k], size=mask.sum(), random_state=rng)
        return samples


class WMMDist(rv_continuous):
    """
    Weibull [2 params] Mixture Model Distribution (WMM) class based on scipy.stats.rv_continuous
    """
    def __init__(self, n_components=2):
        """
        Initialize the WMM distribution.

        Parameters:
        - n_components: int, the number of Weibull components in the mixture
        """
        super().__init__()
        self.name = 'WMM'
        self.n_components = n_components
        self.param_names = ['weights', 'shapes', 'scales']

    def fit(self, x, **kwds):
        """
        Fit the WMM to the data using a simple Expectation-Maximization (EM) algorithm.
        
        Parameters:
        - x: 1D array of data points to fit the WMM
        
        Returns:
        - (weights, shapes, scales): Tuple of fitted parameters
        """
        
        # Parameter for stopping criteria
        tol = kwds['tol'] if 'tol' in kwds else 0.001
        random_state = kwds['random_state'] if 'random_state' in kwds else None
        
        # Initialize the parameters
        rng = np.random.default_rng(random_state)
        weights = np.ones(self.n_components) / self.n_components
        shapes = rng.random(size=self.n_components) + 1
        scales = rng.random(size=self.n_components) + 1
        params0 = np.concatenate([weights, shapes, scales])
        
        # Bounds: weights between 0 and 1, shapes and scales positive
        bounds = [(0, 1)] * self.n_components + [(1e-10, None)] * self.n_components * 2
        
        def neg_log_likelihood(params):
            # Extract parameters
            weights = params[:self.n_components]
            shapes = params[self.n_components:2*self.n_components]
            scales = params[2*self.n_components:]
            
            # Ensure the weights sum to 1
            sum_weights = np.sum(weights)
            sum_weights = sum_weights if np.abs(sum_weights)!=0 else 1e-300 # Small epsilon to avoid division per 0
            weights = weights / sum_weights
            
            # Calculate negative log-likelihood
            log_pdf_values = self.logpdf(x, weights, shapes, scales)
            return -np.sum(log_pdf_values)
        
        # Optimize the parameters using minimize
        try:
            result = minimize(neg_log_likelihood, params0, bounds=bounds, tol=tol)
        except Exception as e:
            raise e
        
        # Extract the fitted parameters
        fitted_params = result.x
        weights = fitted_params[:self.n_components] / np.sum(fitted_params[:self.n_components])  # Normalize weights
        shapes = fitted_params[self.n_components:2*self.n_components]
        scales = fitted_params[2*self.n_components:]
        
        return weights, shapes, scales


    def pdf(self, x, weights, shapes, scales):
        """
        Probability Density Function (PDF) of the WMM.
    
        Parameters:
        - x: array-like, input data
        - weights: array-like, mixture weights
        - shapes: array-like, component shapes
        - scales: array-like, component scales
    
        Returns:
        - pdf_values: array-like, PDF evaluated at x
        """
        
        pdf_values = np.zeros_like(x, dtype=float)
        for k in range(self.n_components):
            pdf_values += weights[k] * weibull_min.pdf(x, c=shapes[k], scale=scales[k])
        return pdf_values
    
    
    def logpdf(self, x, weights, shapes, scales):
        """
        Log of the Probability Density Function (log-PDF) of the WMM.

        Parameters:
        - x: array-like, input data
        - weights: array-like, mixture weights
        - shapes: array-like, component shapes
        - scales: array-like, component scales

        Returns:
        - log_pdf_values: array-like, log-PDF evaluated at x
        """
        # Get the PDF values
        pdf_values = self.pdf(x, weights, shapes, scales)
        
        # Take the logarithm of the PDF values, handle zero values to avoid log(0)
        log_pdf_values = np.log(np.maximum(pdf_values, 1e-300))  # Small epsilon to avoid log(0)
        
        return log_pdf_values


    def cdf(self, x, weights, shapes, scales):
        """
        Cumulative Distribution Function (CDF) of the WMM.

        Parameters:
        - x: array-like, input data
        - weights: array-like, mixture weights
        - shapes: array-like, component shapes
        - scales: array-like, component scales
        
        Returns:
        - cdf_values: array-like, CDF evaluated at x
        """
        
        cdf_values = np.zeros_like(x, dtype=float)
        for k in range(self.n_components):
            cdf_values += weights[k] * weibull_min.cdf(x, c=shapes[k], scale=scales[k])
        return cdf_values

    def rvs(self, weights, shapes, scales, size=1, random_state=None):
        """
        Generate random samples from the WMM.

        Parameters:
        - weights: array-like, mixture weights
        - shapes: array-like, component shapes
        - scales: array-like, component scales
        - size: int, number of samples to generate
        - random_state: int, seed for random number generator

        Returns:
        - samples: array-like, generated random samples
        """
        
        rng = np.random.default_rng(random_state)
        samples = np.zeros(size)
        component_samples = rng.choice(self.n_components, size=size, p=weights)
        for k in range(self.n_components):
            mask = component_samples == k
            samples[mask] = weibull_min.rvs(c=shapes[k], scale=scales[k], size=mask.sum(), random_state=rng)
        return samples



class KDEGaussian(rv_continuous):
    """
    KDE Gaussian Distribution (KDEGaussian)
    """
    def __init__(self, bw_method='scott'):
        super().__init__()
        self.name = 'KDE'
        self.bw_method = bw_method
        self.param_names = ['fitted_kde']
        

    def fit(self, x):
        fitted_kde = gaussian_kde(x, bw_method=self.bw_method)
        return [fitted_kde]
    
    
    def pdf(self, x, fitted_kde):
        return fitted_kde.pdf(x)
    
    
    def logpdf(self, x, fitted_kde):
        return fitted_kde.logpdf(x)
    
    
    def cdf(self, x, fitted_kde):
        cdf_values = [fitted_kde.integrate_box_1d(-np.Inf, xi) for xi in x]
        return np.array(cdf_values)
    
    
    def rvs(self, fitted_kde, size=1, random_state=None):
        samples = fitted_kde.resample(size=size, seed=random_state)
        return samples.flatten()


       
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Example usage of GMMDist
    # Generate synthetic data
    data = np.concatenate([
        np.random.normal(0, 1, size=3000),
        np.random.normal(5, 0.5, size=2000),
    ])
    
    # Instantiate and fit the GMM
    distribution = GMMDist(n_components=2)
    params = distribution.fit(data, max_iter=200) #max_iter is optional
    print('GMM Params', params)
    
    # Plot the fitted GMM PDF
    x = np.linspace(min(data), max(data), 1000)
    pdf = distribution.pdf(x, *params)
    plt.hist(data, bins=30, density=True, alpha=0.5, label="Data")
    plt.plot(x, pdf, label="Fitted GMM", color='red')
    plt.legend()
    plt.show()    
    # Plot generated random samples
    samples = distribution.rvs(*params, size=10000)
    plt.hist(samples, bins=30, density=True, alpha=0.5, label="GMM random samples")
    plt.plot(x, pdf, label="Fitted GMM", color='red')
    plt.legend()
    plt.show()
    
    log_likelihood = np.sum(distribution.logpdf(data, *params))
    print('GMM Log_likelihood = ', log_likelihood)
    
    
    
    # Example usage of WMMDist
    # Generate synthetic data
    data = np.concatenate([
        weibull_min.rvs(c=1.5, scale=2, size=30000),
        weibull_min.rvs(c=5, scale=1, size=20000)
    ])
    
    # Instantiate and fit the WMM
    distribution = WMMDist(n_components=2)
    params = distribution.fit(data)
    print('WMM Params', params)
    
    # Plot the fitted GMM PDF
    x = np.linspace(min(data), max(data), 1000)
    pdf = distribution.pdf(x, *params)
    plt.hist(data, bins=30, density=True, alpha=0.5, label="Data")
    plt.plot(x, pdf, label="Fitted WMM", color='red')
    plt.legend()
    plt.show()    
    # Plot generated random samples
    samples = distribution.rvs(*params, size=10000)
    plt.hist(samples, bins=30, density=True, alpha=0.5, label="WMM random samples")
    plt.plot(x, pdf, label="Fitted WMM", color='red')
    plt.legend()
    plt.show()
    
    log_likelihood = np.sum(distribution.logpdf(data, *params))
    print('WMM Log_likelihood = ', log_likelihood)
    
    
    
    # Example usage of KDEGaussian
    # Generate synthetic data
    #np.random.seed(0)
    data = np.random.normal(loc=0, scale=1, size=10000)
    
    # Instantiate and fit the GMM
    distribution = KDEGaussian()
    params = distribution.fit(data)
    
    # Plot the fitted KDEGaussian PDF
    x = np.linspace(min(data), max(data), 1000)
    pdf = distribution.pdf(x, *params)
    plt.hist(data, bins=30, density=True, alpha=0.5, label="Data")
    plt.plot(x, pdf, label="Fitted KDEGaussian", color='red')
    plt.legend()
    plt.show()    
    # Plot generated random samples
    samples = distribution.rvs(*params, size=500)
    plt.hist(samples, bins=30, density=True, alpha=0.5, label="KDEGaussian random samples")
    plt.plot(x, pdf, label="Fitted KDEGaussian", color='red')
    plt.legend()
    plt.show()
    
    log_likelihood = np.sum(distribution.logpdf(data, *params))
    print('KDE Log_likelihood = ', log_likelihood)