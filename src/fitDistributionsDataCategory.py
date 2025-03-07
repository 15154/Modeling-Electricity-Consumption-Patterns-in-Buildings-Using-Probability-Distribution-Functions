#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:30:47 2024

@author: veroneze
"""

import pandas as pd
import scipy.stats as stats
import numpy as np
import itertools
import timeit
from sys import argv
import utils
from mixture_models_kde import GMMDist, WMMDist, KDEGaussian


# These variables will be received in the call: ###
dataset_set = argv[1]
dataset_i = argv[2]
op_filtro = argv[3] # DH or MDH or SDH
# dataset_set = 'REC2'
# dataset_i = '0'
# op_filtro = 'DH' # DH or MDH or SDH
print(dataset_set, dataset_i, op_filtro)
###################################################
# Change these variables if necessary:
target_column = 'next_hour_cons'
n_mc_samples = 9999 # 9999 number of Monte Carlo samples  in the GoF test
random_state = None


class distribution_info():
    def __init__(self, name, dist, compare_with=None):
        self.name = name
        self.dist = dist
        self.compare_with = compare_with

# Wrong information here will lead to wrong results!
dists_info = []
dists_info.append(distribution_info('norm', stats.norm))
dists_info.append(distribution_info('lognorm_2', stats.lognorm))
dists_info.append(distribution_info('lognorm', stats.lognorm, 'lognorm_2'))
dists_info.append(distribution_info('exponnorm', stats.exponnorm))
dists_info.append(distribution_info('invgauss_2', stats.invgauss))
dists_info.append(distribution_info('invgauss', stats.invgauss, 'invgauss_2'))
dists_info.append(distribution_info('beta', stats.beta))
dists_info.append(distribution_info('gamma_2', stats.gamma))
dists_info.append(distribution_info('gamma', stats.gamma, 'gamma_2'))
dists_info.append(distribution_info('gumbel_r', stats.gumbel_r))
dists_info.append(distribution_info('rayleigh_1', stats.rayleigh))
dists_info.append(distribution_info('rayleigh', stats.rayleigh, 'rayleigh_1'))
dists_info.append(distribution_info('weibull_min_2', stats.weibull_min))
dists_info.append(distribution_info('weibull_min', stats.weibull_min, 'weibull_min_2'))
dists_info.append(distribution_info('GMM2', GMMDist(n_components=2), 'norm'))
dists_info.append(distribution_info('GMM3', GMMDist(n_components=3), 'GMM2'))
dists_info.append(distribution_info('WMM2', WMMDist(n_components=2), 'weibull_min_2'))
dists_info.append(distribution_info('WMM3', WMMDist(n_components=3), 'WMM2'))
dists_info.append(distribution_info('KDEscott', KDEGaussian(bw_method='scott')))
dists_info.append(distribution_info('KDEsilverman', KDEGaussian(bw_method='silverman')))

complex_dist = [d.name for d in dists_info if d.name[0:3] in ['GMM', 'WMM', 'KDE']]

dist_param_names = {}
for d in dists_info:
    if d.name not in complex_dist:
        dist_param_names[d.name] = utils.get_param_names(d.dist)
    else:
        dist_param_names[d.name] = d.dist.param_names


if op_filtro == "DH":
    filtro = ['Day', 'Hour']
elif op_filtro == "SDH":
    filtro = ['Season', 'Day', 'Hour']
else:
    filtro = ['Month', 'Day', 'Hour']

# Load data
filename = '../datasets/' + dataset_set + '/' + dataset_i + '.csv'
df = pd.read_csv(filename, index_col="Datetime", usecols=lambda col: col not in ['Unnamed: 0', 'Unnamed: 0.1'])

# List to store the results
results = []

# Get unique values of each column specified in the 'filtro' list
unique_values = [np.sort(df[col].unique()) for col in filtro]

# Loop through all combinations of the unique values of the specified columns
start_time = timeit.default_timer()
for combination in itertools.product(*unique_values):
    print(combination)
    
    # Create filter condition based on the combination
    condition = True
    for col, value in zip(filtro, combination):
        condition &= (df[col] == value)
        
    # Filter the dataframe using the condition
    filtered_df = df[condition]
    data = filtered_df[target_column]

    # Fit each distribution and compute the metrics
    dsucess = {}
    dnparams = {}
    dll = {}
    for dinfo in dists_info:
        dist_name = dinfo.name
        distribution = dinfo.dist
        print(dist_name)
        
        llr_pvalue = np.NaN
        sucess,nparams,fit_params,log_likelihood,statistic,pvalue,aic,bic = utils.fit_evaluate_distribution(distribution, dist_name, dist_param_names[dist_name], data, n_mc_samples=n_mc_samples, random_state=random_state)
        dsucess[dist_name] = sucess
        dnparams[dist_name] = nparams
        dll[dist_name] = log_likelihood
        
        if dinfo.compare_with is not None and sucess == 1:
            if dsucess[dinfo.compare_with] == 1:
                nparam_null = dnparams[dinfo.compare_with]
                loglik_null = dll[dinfo.compare_with]
                nparam_alt = nparams
                loglik_alt = log_likelihood
                _, llr_pvalue = utils.likelihood_ratio_test(loglik_null, loglik_alt, nparam_null, nparam_alt)
            else:
                llr_pvalue = -np.Inf
        
        results.append([
                    combination,
                    data.shape[0],
                    dist_name,
                    sucess,
                    nparams,
                    fit_params,
                    log_likelihood,
                    statistic,
                    pvalue,
                    aic,
                    bic,
                    llr_pvalue])
    # print('time running(s): ', timeit.default_timer() - start_time) 

stop_time = timeit.default_timer()
print('runtime(s): ', stop_time - start_time)
df_results = pd.DataFrame(results, columns=['combination', 'n', 'distribution', 'success', 'nparams', 'fit_params', 'log_likelihood', 'ks_statistic', 'ks_pvalue', 'aic', 'bic', 'llr'])
df_results.to_csv('../results/'+dataset_set+'/'+dataset_i+'_'+op_filtro+'.csv', index=False)