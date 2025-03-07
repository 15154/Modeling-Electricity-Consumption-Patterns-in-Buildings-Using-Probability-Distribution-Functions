#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 16:58:43 2024

@author: veroneze
"""

import pandas as pd
import scipy.stats as stats
import numpy as np
import itertools
import timeit
from sys import argv
import os
import utils
from mixture_models_kde import GMMDist, WMMDist, KDEGaussian


# These variables will be received in the call: ###
dataset_set = argv[1]
dataset_i = argv[2]
op_filtro = argv[3] # DH or MDH or SDH
dist_name = argv[4]
# dataset_set = 'REC2'
# dataset_i = '0'
# op_filtro = 'DH' # DH or MDH or SDH
# dist_name = 'GMM2'
print(dataset_set, dataset_i, op_filtro, dist_name)
###################################################
# Change these variables if necessary:
target_column = 'next_hour_cons'
n_mc_samples = 9999 # 9999 number of Monte Carlo samples  in the GoF test
random_state = None



df_dists = pd.read_csv('distributions.csv', index_col=0)
distribution = eval(df_dists.loc[dist_name, 'distribution'])


if  dist_name[0:3] not in ['GMM', 'WMM', 'KDE']:
    dist_param_names = utils.get_param_names(distribution)
else:
    dist_param_names = distribution.param_names


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
    
    
    sucess,nparams,fit_params,log_likelihood,statistic,pvalue,aic,bic = utils.fit_evaluate_distribution(distribution, dist_name, dist_param_names, data, n_mc_samples=n_mc_samples, random_state=random_state)
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
                bic])

stop_time = timeit.default_timer()
print('runtime(s): ', stop_time - start_time)

df_results = pd.DataFrame(results, columns=['combination', 'n', 'distribution', 'success', 'nparams', 'fit_params', 'log_likelihood', 'ks_statistic', 'ks_pvalue', 'aic', 'bic'])
path_save = '../results/'+dataset_set+'/dists/'
os.makedirs(path_save, exist_ok=True)
df_results.to_csv(path_save+dataset_i+'_'+op_filtro+'_'+dist_name+'.csv', index=False, mode='a')