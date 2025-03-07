#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:47:19 2024

@author: veroneze
"""


import numpy as np
import scipy.stats as stats


# Function to count instances and calculate mean and std of target column
def aggregate_combinations(df, columns, target):
    """
    Function to count instances, and calculate mean and standard deviation of target column
    for each combination of values in the specified columns.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    columns (list): A list of column names to group by.

    Returns:
    pd.DataFrame: A DataFrame with count, mean, and standard deviation for each combination.
    """
    # Group by the specified columns and apply multiple aggregations
    grouped = df.groupby(columns).agg(
        Count=(target, 'size'),  # Count the number of occurrences
        Mean_Target=(target, 'mean'),  # Calculate the mean of the target column
        Std_Target=(target, 'std')  # Calculate the standard deviation of the target column
    ).reset_index()
    
    return grouped


# Function to filter data based on dictionary of column-value pairs
def filter_dataframe(df, filter_dict):
    # Apply filter for each column based on the dictionary key-value pairs
    for column, value in filter_dict.items():
        if column in df.columns:
            df = df[df[column] == value]
        else:
            print(f"Column '{column}' not found in the DataFrame.")
    return df


# Function to get the names of the parameters of a scipy.stats distribution
def get_param_names(distribution):
    param_names = distribution.shapes.split(', ') if distribution.shapes else []
    param_names += ['loc', 'scale']
    return param_names


# Function to compute AIC and BIC
def compute_aic_bic(log_likelihood, num_params, n):
    """
    Function to compute AIC and BIC criteria

    Parameters:
    log_likelihood: log_likelihood value
    num_params: number of parameters estimated by the model
    n: number of data points

    Returns:
    aic and bic values
    """
    aic = 2 * num_params - 2 * log_likelihood
    bic = np.log(n) * num_params - 2 * log_likelihood
    return aic, bic


# Likelihood-Ratio Test (LRT) calculation
# https://en.wikipedia.org/wiki/Likelihood-ratio_test
# https://en.wikipedia.org/wiki/Wilks%27_theorem
def likelihood_ratio_test(loglik_null, loglik_alt, nparam_null, nparam_alt):
    degrees_of_freedom = nparam_alt -  nparam_null
    lr_statistic = 2 * (loglik_alt - loglik_null)
    p_value_lrt = stats.chi2.sf(lr_statistic, df=degrees_of_freedom)
    return lr_statistic, p_value_lrt


"""
Refs:
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.goodness_of_fit.html
- Anthony Zeimbekakis, Elizabeth D. Schifano & Jun Yan (2024) On Misuses of the Kolmogorov–Smirnov Test for One-Sample Goodness-of-Fit, The American Statistician, 78:4,
481-487, DOI: 10.1080/00031305.2024.2356095.
- Anthony Zeimbekakis (2022) On Misuses of the Kolmogorov–Smirnov Test for One-Sample Goodness-of-Fit, Thesis at Department of Statistics, University of Connecticut.
- https://www.clayford.net/statistics/parametric-bootstrap-of-kolmogorov-smirnov-test/
"""   
def ks_test_parametric_bootstrap(data, dist, fit_params, n_mc_samples=9999, random_state=None):
    """
    Performs a Kolmogorov-Smirnov goodness-of-fit test with parametric bootstrap.

    Args:
        data: The observed data.
        dist: Distribution to test.
        params: The estimated parameters of the distribution.

    Returns:
        The test statistic and the p-value.
    """
    
    rng = np.random.default_rng(random_state)
    
    n = len(data)

    # Calculate the observed test statistic
    d_obs = stats.ks_1samp(data, dist.cdf, args=fit_params)[0]

    # Generate bootstrap replicates
    d_boot = np.zeros(n_mc_samples)
    for b in range(n_mc_samples):
        # Sample from the fitted distribution
        bootstrap_sample = dist.rvs(size=n, *fit_params, random_state=rng)
        
        try:
            # Fit the distribution to the bootstrap sample
            if dist.name in ['GMM', 'WMM']:
                bootstrap_params = dist.fit(bootstrap_sample, random_state=random_state)
            else:
                bootstrap_params = dist.fit(bootstrap_sample)
            # Calculate the bootstrap test statistic
            d_boot[b] = stats.ks_1samp(bootstrap_sample, dist.cdf, args=bootstrap_params)[0]
        except Exception as e:
            print(f"Error fitting in ks_test_parametric_bootstrap - {dist.name}: {e}")
    
    # Calculate the p-value
    # p_value = np.mean(d_boot >= d_obs)
    p_value = (np.sum(d_boot >= d_obs) + 1) / (n_mc_samples  + 1) # From scipy.stats.goodness_of_fit

    return d_obs, p_value


def fit_evaluate_distribution(distribution, dist_name, param_names, data, n_mc_samples=9999, random_state=None):
    simple_dist = True
    if dist_name[:3] in ['GMM', 'WMM', 'KDE']:
        simple_dist = False
    
    n_data = len(data)
    
    nparams = 0
    log_likelihood = 0
    statistic = 0
    pvalue = 0
    aic = 0
    bic = 0
    
    try:
        # Fit the distribution to the data
        if dist_name in ['lognorm_2', 'invgauss_2', 'gamma_2', 'rayleigh_1', 'weibull_min_2']:
            n_fixed_params = 1
            params = distribution.fit(data, floc=0)
        elif dist_name == 'beta_2':
            n_fixed_params = 2
            params = distribution.fit(data, floc=0, fscale=1)
        elif dist_name[:3] in ['GMM', 'WMM']:
            n_fixed_params = 0
            params = distribution.fit(data, random_state=random_state)
        else:
            n_fixed_params = 0
            params = distribution.fit(data)
        sucess=1
    except Exception as e:
        print(f"Error fitting {dist_name}: {e}")
        sucess = 0
        return sucess,nparams,{},log_likelihood,statistic,pvalue,aic,bic
    
    # Create dictionary with the names and values of the parameters
    fit_params = {name: param for name, param in zip(param_names, params)}
    
    if dist_name[:3]!='KDE' and ((np.isinf(params)).any() or (np.isnan(params)).any()):
        sucess = -1
        return sucess,nparams,fit_params,log_likelihood,statistic,pvalue,aic,bic

    # Get the log likelihood (sum of logpdf values)
    # log_likelihood = np.sum(distribution.logpdf(data, *params)) # has -Inf values when pdfvalue = 0
    pdf_values = distribution.pdf(data, *params)
    if np.all(pdf_values == 0):
        sucess = -3
        return sucess,nparams,fit_params,log_likelihood,statistic,pvalue,aic,bic
    # Take the logarithm of the PDF values, handle zero values to avoid log(0)
    log_pdf_values = np.log(np.maximum(pdf_values, 1e-300))  # Small epsilon to avoid log(0)
    log_likelihood = np.sum(log_pdf_values)
    if np.isnan(log_likelihood) or np.isneginf(log_likelihood) or np.isinf(log_likelihood):
        sucess = -2
        return sucess,nparams,fit_params,log_likelihood,statistic,pvalue,aic,bic
    
    # Calculate AIC and BIC
    nparams = len(np.array(params).flatten()) - n_fixed_params
    aic, bic = compute_aic_bic(log_likelihood, nparams, n_data)
    
    # Calculate statistic and pvalue
    if simple_dist:
        gof = stats.goodness_of_fit(distribution, data, fit_params=fit_params, statistic='ks', n_mc_samples=n_mc_samples, random_state=random_state)
        statistic = gof.statistic
        pvalue = gof.pvalue
    else:
        statistic, pvalue = ks_test_parametric_bootstrap(data, distribution, params, n_mc_samples=n_mc_samples, random_state=random_state)
        
    
    return sucess,nparams,fit_params,log_likelihood,statistic,pvalue,aic,bic
