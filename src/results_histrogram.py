#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 09:27:33 2024

@author: veroneze
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
import re
from scipy import stats
from mixture_models_kde import GMMDist, WMMDist, KDEGaussian
from numpy import array

# set the information bellow:
dataset_set = 'REC2'
dataset_i = 'A'
#filter_dict = {'Month': 2, 'Day': 7, 'Hour': 11}
#filter_dict = {'Season':'winter', 'Day': 4, 'Hour': 3}
filter_dict = {'Day':5, 'Hour':22}
sig_level = 0.05
##############################

if 'Season' in filter_dict.keys():
    cat = 'SDH'
elif 'Month' in filter_dict.keys():
    cat = 'MDH'
else:
    cat = 'DH'
print(cat)


data_names = {}
data_names['UK'] = 'UK'
data_names['REC1'] = 'REC1'
data_names['REC2'] = 'REC2'

# Load csv distributions
df_dists = pd.read_csv('distributions.csv', index_col=0)

# Load results
filename_res = '../results/' + dataset_set + '/' + dataset_i+'_'+cat+'.csv'
df_res = pd.read_csv(filename_res, index_col=2)

# Select results
comb = str(tuple(filter_dict.values()))
#selected_df_res = df_res[(df_res['combination'] == comb) & (df_res['success'] == 1)]
selected_df_res = df_res[(df_res['combination'] == comb) & (df_res['ks_pvalue'] > sig_level)]
selected_df_res= selected_df_res.sort_values('log_likelihood', ascending=True)

# Load data
filename = '../datasets/' + dataset_set + '/' + dataset_i + '.csv'
df = pd.read_csv(filename, index_col="Datetime", usecols=lambda col: col not in ['Unnamed: 0', 'Unnamed: 0.1'])

# Select data
selected_df = utils.filter_dataframe(df, filter_dict)
data = selected_df['next_hour_cons']
print("Number of samples", data.shape[0])


# Plot
bin_method = 'sturges'
#'sturges', 'fd', 'scott', 'doane', 'sqrt'
x = np.linspace(np.min(data), np.max(data), 1000)
plt.style.use('tableau-colorblind10')
plt.figure(figsize=(6.4, 4.8))
hist_info = plt.hist(data, bins=bin_method, label='Data',density=True)
print('Number of bins = ', len(hist_info[0]))
for row in selected_df_res.head().itertuples():
    dist_name = row.Index
    print(dist_name)
    dist_class = eval(df_dists.loc[dist_name, 'distribution'])
    if 'KDE' in dist_name: # as it does not have params, we need to fit it again
        params = dist_class.fit(data)
        pdf_fitted = dist_class.pdf(x, *params)
    else:
        #dist_class = getattr(stats, dist_name)
        params = eval(row.fit_params)
        print(params)
        pdf_fitted = dist_class.pdf(x, **params)    
    if dist_name == 'lognorm_2':
        rotulo = 'lognorm2'
    elif dist_name == 'invgauss_2':
        rotulo = 'invgauss2'
    elif dist_name == 'gamma_2':
        rotulo = 'gamma2'
    elif dist_name == 'gumbel_r':
        rotulo = 'Gumbel'
    elif dist_name == 'rayleigh_1':
        rotulo = 'Rayleigh1'
    elif dist_name == 'rayleigh':
        rotulo = 'Rayleigh'
    elif dist_name == 'weibull_min_2':
        rotulo = 'Weibull2'
    elif dist_name == 'weibull_min':
        rotulo = 'Weibull'
    #elif dist_name == 'KDEsilverman':
    #    rotulo = 'KDEsilv.'
    else:
      rotulo = dist_name    
    plt.plot(x, pdf_fitted, label=rotulo)
plt.title(data_names[dataset_set]+' - '+dataset_i+' - ' + re.sub(r"[{}']","", str(filter_dict)), fontweight='bold')
plt.xlabel('x', fontweight='bold')
plt.ylabel('Density', fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
#plt.xlim(right=0.1)
#plt.ylim(top=1.2)
plt.legend()
plt.grid()
plt.savefig('fig.png', dpi=300, bbox_inches='tight')
plt.show()
