#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:18:56 2024

@author: veroneze
"""

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')


def boxplot_dists(df_boxplot_d, column, title, distribution_order, df_sucess, df_pvalue, df_llr, df_success_pvalue):
    output_file = '../results/figs/' + title + '_' + column + '.png'

    # Create the boxplot
    plt.figure(figsize=(14, 7))
    ax = sns.boxplot(data=df_boxplot_d, x='distribution', y=column, hue='filtro', width=0.8, showfliers=False)
    
    # Customize the plot
    plt.title(title+' - '+column, fontsize=10, fontweight='bold')
    plt.xlabel('', fontsize=1)
    plt.ylabel('', fontsize=1, fontweight='bold')
    plt.legend(title='', fontsize=10)
    
    
    # Get current xticks and labels
    xticks = ax.get_xticks()
    xtick_labels = [label.get_text() for label in ax.get_xticklabels()]
    
    # Replace specific labels
    for i, label in enumerate(xtick_labels):
        if label == 'lognorm_2':
            xtick_labels[i] = 'lognorm2'
        elif label == 'invgauss_2':
            xtick_labels[i] = 'invgauss2'
        elif label == 'gamma_2':
            xtick_labels[i] = 'gamma2'
        elif label == 'gumbel_r':
            xtick_labels[i] = 'Gumbel'
        elif label == 'rayleigh_1':
            xtick_labels[i] = 'Rayleigh1'
        elif label == 'rayleigh':
            xtick_labels[i] = 'Rayleigh'
        elif label == 'weibull_min_2':
            xtick_labels[i] = 'Weibull2'
        elif label == 'weibull_min':
            xtick_labels[i] = 'Weibull'
        elif label == 'KDEsilverman':
            xtick_labels[i] = 'KDEsilv.'
    # Update xtick labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontweight='bold')
    
    
    #plt.xticks(rotation=45, fontweight='bold')
    plt.yticks(fontweight='bold')
    
    ylim_max = df_boxplot_d[column].max()
    ylim_min = df_boxplot_d[column].min()
    #plt.ylim(bottom=max(-1500,ylim_min), top=max(2*ylim_max,200))
    if "REC1" in title and column=='log_likelihood':
        plt.ylim(bottom=-1500, top=1050)
        y_coord = ylim_max + 5
    elif "REC1" in title and column=='aic':
        plt.ylim(bottom=-1650, top=1000)
        y_coord = ylim_min - 300
    elif "UK" in title and column=='log_likelihood':
        plt.ylim(bottom=-1000, top=3400)
        y_coord = ylim_max + 5
        if y_coord > 3000: y_coord -= 500
    elif "UK" in title and column=='aic':
        plt.ylim(bottom=-7000, top=1100)
        y_coord = ylim_min - 1300
        if y_coord < -6500: y_coord += 600
    elif "REC2" in title and column=='log_likelihood':
        plt.ylim(bottom=-4000, top=1400)
        y_coord = ylim_max + 5
        if y_coord > 1800: y_coord -= 3000
    elif "REC2" in title and column=='aic':
        plt.ylim(bottom=-2500, top=6000)
        y_coord = ylim_min - 1250
        if y_coord < -2500: y_coord += 5900
    else:
        y_coord = ylim_max + 5 if column=='log_likelihood' else ylim_min - 200
    #plt.ylim(bottom=-1500, top=2000) #REC2 loglike
    
    if column == 'log_likelihood' or column == 'aic':
        # Add extra information on top of each box
        for i, distribution in enumerate(distribution_order):
            t_sucess = ''
            t_ks = ''
            t_llr = ''
            t_success_ks = ''
            for cat in cats:
                if (dataset_name, cat, distribution)  in df_sucess.index:
                    p_success = df_sucess.loc[(dataset_name, cat, distribution), 'success_percentage']
                else:
                    p_success = -1
                t_sucess += f"{p_success:5.1f}" + ' '
                
                if (dataset_name, cat, distribution)  in df_pvalue.index:
                    p_ks = df_pvalue.loc[(dataset_name, cat, distribution), 'pvalue_percentage']
                else:
                    p_ks = -1
                t_ks += f"{p_ks:5.1f}" + ' '
                
                if distribution in ['lognorm','invgauss','gamma','rayleigh','weibull_min','GMM2','GMM3','WMM2','WMM3'] and (dataset_name, cat, distribution) in df_llr.index:
                    p_llr = df_llr.loc[(dataset_name, cat, distribution), 'llr_percentage']
                    t_llr += f"{p_llr:5.1f}" + ' '                    
                    
                if (dataset_name, cat, distribution)  in df_success_pvalue.index:
                    p_success_ks = df_success_pvalue.loc[(dataset_name, cat, distribution), '2success_percentage']
                else:
                    p_success_ks = -1
                t_success_ks += f"{p_success_ks:.1f}" + ' '
                    
            t_sucess = t_sucess[:-1]
            t_ks = t_ks[:-1]
            t_llr = t_llr[:-1]
            t_success_ks = t_success_ks[:-1]
            plt.text(i,
                      y_coord,
                      t_success_ks, #t_sucess+'\n'+t_ks+'\n'+t_llr,
                      ha='center',
                      va='bottom',
                      fontfamily='monospace',
                      fontsize=10,#7,
                      fontweight='bold',
                      color='darkred',
                      rotation=45)
    
    tick_positions = ax.get_xticks() # Get the tick positions for each distribution
    # Draw vertical lines between distributions
    for pos in tick_positions[:-1]:  # Exclude the last tick to avoid a line after the final distribution
        plt.axvline(x=pos + 0.5, color='lightgray', linestyle='--', linewidth=1)
    
    #plt.tight_layout()
    #plt.show()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')


# Pivot the table to get the desired format
def pivot_results(dataframe, values, cols_order):
    dataframe = dataframe.pivot_table(
        index=['dataset_name', 'filtro'],
        columns='distribution',
        values=values
    ).reset_index()
    dataframe = dataframe[cols_order]
    return dataframe


def ranking(df):
    # Set the columns containing the distributions
    distribution_columns = df.columns[2:]  # All columns after 'dataset_name' and 'filtro'
    
    # Create a new dataframe to store the rankings
    df_rankings = df.copy()
    
    # Compute the rankings for each row
    df_rankings[distribution_columns] = df[distribution_columns].rank(axis=1, method='min', numeric_only=True, na_option='top', ascending=False)
    
    return df_rankings



# Set the information below:
dataset_set = 'REC1'
folder_path = '../results/'
significance_level = 0.05
#############################

data_names = {}
data_names['UK'] = 'UK'
data_names['REC1'] = 'REC1'
data_names['REC2'] = 'REC2'


# # I am doing it to keep the same order of the distributions in the output files
# df_dists = pd.read_csv('distributions.csv', index_col=0)
# cols_order = None
# distribution_order = list(df_dists.index)
# cols_order = ['dataset_name', 'filtro'] + distribution_order


results_sucess = []
res_sucess_pvalue = []
results_pvalue = []
results_boxplot = []
results_llr = []
folder_path_res = folder_path+dataset_set
file_list = os.listdir(folder_path_res)
for filename in file_list:
    if 'DH' in filename:
        i_ = filename.rfind('_')
        dataset_name = filename[:i_]
        filtro = filename[i_+1:-4]
        print(filename, dataset_name, filtro)
        
        # Read the csv file with the results
        df = pd.read_csv(folder_path_res + '/' + filename)
        
        # # I am doing it to keep the same order of the distributions in the output files
        # # Some datasets does not have the results for alls dists yet
        distribution_order = list(df[df['combination'] == df['combination'][0]]['distribution'])
        cols_order = ['dataset_name', 'filtro'] + distribution_order
        
        # Compute the % of sucess of each distribution
        percentage_success = (
            df.groupby('distribution')
            .apply(lambda x: (x['success'] == 1).mean() * 100, include_groups=False)
            .reset_index(name='success_percentage')
        )
        percentage_success['dataset_name'] = dataset_name
        percentage_success['filtro'] = filtro
        results_sucess.append(percentage_success)
        
        # Compute the % of (1) sucess AND (2) ks_pvalue > significance_level of each distribution
        perc_success_pvalue = (
            df.groupby('distribution')
            .apply(lambda x: (x['ks_pvalue'] > significance_level).mean() * 100, include_groups=False)
            .reset_index(name='2success_percentage')
        )
        perc_success_pvalue['dataset_name'] = dataset_name
        perc_success_pvalue['filtro'] = filtro
        res_sucess_pvalue.append(perc_success_pvalue)
        
        # Compute the % of pvalues>significance_level of each distribution when sucess==1
        df_filter_sucess = df[df['success'] == 1]
        percentage_pvalue = (
            df_filter_sucess.groupby('distribution')
            .apply(lambda x: (x['ks_pvalue'] > significance_level).mean() * 100, include_groups=False)
            .reset_index(name='pvalue_percentage')
        )
        percentage_pvalue['dataset_name'] = dataset_name
        percentage_pvalue['filtro'] = filtro
        results_pvalue.append(percentage_pvalue)
        
        # Datafram with sucess==1 and ks_pvalue>significance_level
        df_filter_sucess_pvalue = df_filter_sucess[df_filter_sucess['ks_pvalue'] > significance_level]
        df_filter_sucess_pvalue['dataset_name'] = dataset_name
        df_filter_sucess_pvalue['filtro'] = filtro
        results_boxplot.append(df_filter_sucess_pvalue)
        
        # Log-likelihood ratio test
        # Compute the % of llr>significance_level of each distribution when sucess==1 and ks_pvalue>significance_level
        percentage_llr = (
            df_filter_sucess_pvalue.groupby('distribution')
            .apply(lambda x: (x['llr'] <= significance_level).mean() * 100, include_groups=False)
            .reset_index(name='llr_percentage')
        )
        percentage_llr['dataset_name'] = dataset_name
        percentage_llr['filtro'] = filtro
        results_llr.append(percentage_llr)

df_success = pd.concat(results_sucess, ignore_index=True)
df_success_p = pivot_results(df_success, 'success_percentage', cols_order)
df_success_p.to_csv(folder_path+dataset_set+'_success.csv', index=False)
df_success_r = ranking(df_success_p)
df_success_r.to_csv(folder_path+dataset_set+'_success_rank.csv', index=False)
df_success = df_success.set_index(['dataset_name','filtro','distribution'])

df_success_pvalue = pd.concat(res_sucess_pvalue, ignore_index=True)
df_success_pvalue_p = pivot_results(df_success_pvalue, '2success_percentage', cols_order)
df_success_pvalue_p.to_csv(folder_path+dataset_set+'_2success.csv', index=False)
df_success_pvalue_r = ranking(df_success_pvalue_p)
df_success_pvalue_r.to_csv(folder_path+dataset_set+'_2success_rank.csv', index=False)
df_success_pvalue = df_success_pvalue.set_index(['dataset_name','filtro','distribution'])

df_pvalue = pd.concat(results_pvalue, ignore_index=True)
df_pvalue_p = pivot_results(df_pvalue, 'pvalue_percentage', cols_order)
df_pvalue_p.to_csv(folder_path+dataset_set+'_pvalue.csv', index=False)
df_pvalue_r = ranking(df_pvalue_p)
df_pvalue_r.to_csv(folder_path+dataset_set+'_pvalue_rank.csv', index=False)
df_pvalue = df_pvalue.set_index(['dataset_name','filtro','distribution'])

df_llr = pd.concat(results_llr, ignore_index=True)
df_llr_p = pivot_results(df_llr, 'llr_percentage', cols_order)
df_llr_p.to_csv(folder_path+dataset_set+'_llr.csv', index=False)
df_llr = df_llr.set_index(['dataset_name','filtro','distribution'])

# --- Boxplots for when sucess==1 and ks_pvalue>significance_level ---
# Sort data by the predefined distribution order and filter for only those distributions
df_boxplot = pd.concat(results_boxplot, ignore_index=True)
cats = ['DH', 'SDH', 'MDH'] if dataset_set != 'REC1' else ['DH', 'SDH']
df_boxplot['distribution'] = pd.Categorical(df_boxplot['distribution'], categories=distribution_order, ordered=True)
df_boxplot['filtro'] = pd.Categorical(df_boxplot['filtro'], categories=cats, ordered=True)
df_boxplot = df_boxplot.sort_values(['distribution', 'filtro'])

for dataset_name in df_boxplot['dataset_name'].unique():
    title = data_names[dataset_set] + ' ' + dataset_name
    df_boxplot_d = df_boxplot[df_boxplot['dataset_name'] == dataset_name]
    boxplot_dists(df_boxplot_d, 'log_likelihood', title, distribution_order, df_success, df_pvalue, df_llr, df_success_pvalue)
    boxplot_dists(df_boxplot_d, 'aic', title, distribution_order, df_success, df_pvalue, df_llr, df_success_pvalue)
