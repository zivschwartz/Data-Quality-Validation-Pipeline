#!/usr/bin/env python
# coding: utf-8

# In[80]:


import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt


'''
@description: this function reads in a series of filenames and separates them into clean + dirty filenames
@params: all_files - the filename of all batch .csv files, should be of the form 'filepath/*.csv'
@return: clean_files - list, sorted list of all clean data filenames
         dirty_files - list, sorted list of all dirty data filenames
'''
def partition_data_files(all_files):
    data = glob.glob(all_files)
    dirty_files = []
    clean_files = []
    
    for file in data:
        if 'dirty' in file:
            dirty_files.append(file)
        else:
            clean_files.append(file)
            
    #sort in place with our custom sort function (written below)
    dirty_files.sort(key=filename_num) 
    clean_files.sort(key=filename_num)
    
    return clean_files, dirty_files

'''
@description: used in partition_data_files; helper function for .sort(key=____)
              bc Python doesn't sort the filename strings in the order we want by default
@params: filename - the single filename we want the number (integer order) of
@return: order_number - the number of the file
'''
def filename_num(filename):
    remove_csv = filename.split('.')[0]
    number = remove_csv.split('_')[-1]
    return int(number)


'''
@description: this function generates the completeness ratio dataframes for both clean and dirty data files
@params: clean_files - list of strings, clean filenames
         dirty_files - list of strings, dirty filenames
@return: clean_completeness_ratio_df - dataframe of the ratio of nulls to non-nulls in the clean dataframe
         dirty_completeness_ratio_df - dataframe of the ratio of nulls to non-nulls in the dirty dataframe
'''
def completeness_dataframes(clean_files, dirty_files):    
    clean_completeness_ratio_df = pd.DataFrame()
    dirty_completeness_ratio_df = pd.DataFrame()
    
    for file in clean_files:
        data = pd.read_csv(file)

        #get null_counts for all columns
        null_counts = data.isnull().sum()
        #get not null_counts for all columns
        not_null_counts = data.shape[0] - null_counts

        #get ratio
        ratio = null_counts/not_null_counts

        #add to df
        clean_completeness_ratio_df[file] = ratio
        
    for file in dirty_files:
        data = pd.read_csv(file)

        #get null_counts for all columns
        null_counts = data.isnull().sum()
        #get not null_counts for all columns
        not_null_counts = data.shape[0] - null_counts

        #get ratio
        ratio = null_counts/not_null_counts

        #add to df
        dirty_completeness_ratio_df[file] = ratio
    
    data = pd.read_csv(file)
    cols_to_use = data.columns
   
    clean_completeness_ratio_df = clean_completeness_ratio_df[clean_completeness_ratio_df.index.isin(cols_to_use)]
    dirty_completeness_ratio_df = dirty_completeness_ratio_df[dirty_completeness_ratio_df.index.isin(cols_to_use)]
    
        
    return clean_completeness_ratio_df, dirty_completeness_ratio_df


'''
@description: this function generates the distinct counts dataframes for both clean and dirty data files
@params: clean_files - list of strings, clean filenames
         dirty_files - list of strings, dirty filenames
@return: clean_distinct_counts_df - dataframe of the ratio of nulls to non-nulls in the clean dataframe
         dirty_distinct_counts_df - dataframe of the ratio of nulls to non-nulls in the dirty dataframe
'''
    
def distinct_counts_dataframes(clean_files, dirty_files):
    clean_distinct_counts_df = pd.DataFrame()
    dirty_distinct_counts_df = pd.DataFrame()
    
    for file in clean_files:
        data = pd.read_csv(file)
        clean_distinct_counts_df[file] = data.nunique()
    for file in dirty_files:
        data = pd.read_csv(file)
        dirty_distinct_counts_df[file] = data.nunique()
    
    return clean_distinct_counts_df, dirty_distinct_counts_df




# for column in flights_dirty_completeness_ratio_df.index.tolist():
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,5))
#     fig.suptitle(column)
#     ax1.title.set_text('Completeness')
#     ax2.title.set_text('Distinct Values')
#     ax1.plot(np.arange(flights_clean_completeness_ratio_df.shape[1]), flights_clean_completeness_ratio_df.loc[column].values, label = 'clean')
#     ax1.plot(np.arange(flights_dirty_completeness_ratio_df.shape[1]), flights_dirty_completeness_ratio_df.loc[column].values, label = 'dirty')
#     ax2.plot(np.arange(flights_clean_distinct_counts_df.shape[1]), flights_clean_distinct_counts_df.loc[column].values, label = 'clean')
#     ax2.plot(np.arange(flights_dirty_distinct_counts_df.shape[1]), flights_dirty_distinct_counts_df.loc[column].values, label = 'dirty')
#     ax1.legend()
#     ax2.legend()
#     plt.show()

def determine_acceptable_metric_range(batches):
    completeness_df, __ = completeness_dataframes(batches, batches)
    distinct_df, __ = distinct_counts_dataframes(batches, batches)
    completeness_means = completeness_df.mean(axis=1)
    completeness_vars = np.sqrt(completeness_df.var(axis = 1, ddof=len(batches)-1))
    completeness_ste = completeness_df.std(axis=1)
    
    completeness_range = pd.DataFrame(columns = ['min', 'max'])
    completeness_range['min'] = completeness_means - 2*completeness_vars
    completeness_range['max'] = completeness_means + 2*completeness_vars
#     completeness_range['min'] = completeness_means - 3*completeness_ste
#     completeness_range['max'] = completeness_means + 3*completeness_ste
    
    distinct_means = distinct_df.mean(axis=1)
    distinct_vars = np.sqrt(distinct_df.var(axis = 1, ddof=len(batches)-1))
    distinct_ste = distinct_df.std(axis=1)
    
    distinct_range = pd.DataFrame(columns = ['min', 'max'])
    distinct_range['min'] = distinct_means - 2*distinct_vars
    distinct_range['max'] = distinct_means + 2*distinct_vars
#     distinct_range['min'] = distinct_means - 3*distinct_ste
#     distinct_range['max'] = distinct_means + 3*distinct_ste
    return completeness_range, distinct_range

def is_acceptable(train_batch, test_batch):
    completeness_range, distinct_range = determine_acceptable_metric_range(train_batch)
    test_batch_completeness, __ = completeness_dataframes(test_batch, test_batch)
    test_batch_distinct, __ = distinct_counts_dataframes(test_batch, test_batch)
    completeness_within_range = 0
    distinct_within_range = 0
    for i in range(len(test_batch_distinct.values)):
        if test_batch_distinct.values[i][0] >= (distinct_range[i:i+1]['min'][0]) and test_batch_distinct.values[i][0] <= (distinct_range[i:i+1]['max'][0]):
            distinct_within_range = distinct_within_range + 1
        if test_batch_completeness.values[i][0] >= (completeness_range[i:i+1]['min'][0]) and test_batch_completeness.values[i][0] <= (completeness_range[i:i+1]['max'][0]):
            completeness_within_range = completeness_within_range + 1
    
    
    if completeness_within_range/test_batch_completeness.shape[0] > .8 and distinct_within_range/test_batch_completeness.shape[0] > .8:
        return True
    else:
        return False
    
def analysis(i, train_type, clean, dirty, batch_size):
    if train_type == 'rolling':
        dirty_val = is_acceptable(clean[i:i+batch_size], dirty[i+batch_size: i+batch_size+1])
        clean_val = is_acceptable(clean[i:i+batch_size], clean[i+batch_size: i+batch_size+1])
    else:
        dirty_val = is_acceptable(clean[0:i+batch_size], dirty[i+batch_size: i+batch_size+1])
        clean_val = is_acceptable(clean[0:i+batch_size], clean[i+batch_size: i+batch_size+1])
    if dirty_val == False:
        dirty_correct = True
    else:
        dirty_correct = False
    if clean_val== True:
        clean_correct = True
    else:
        clean_correct = False
    row = [train_type, batch_size, i, clean_correct, dirty_correct]
    return row  

def plot_batch(data_name, dataset, batch_size_range):
    df_valid = dataset[dataset.test_batch.isnull() == False]
    
    for train_type in ['rolling', 'increasing']:
        df = df_valid[df_valid.train_type == train_type]
        for batch_size in batch_size_range:
            data = df[df.batch_size == batch_size]
            plt.figure(figsize=(15, 2))
            plt.plot(data.test_batch+batch_size, data.clean_correct, color = 'lightsteelblue', label='_nolegend_')
            plt.scatter(data.test_batch+batch_size, data.clean_correct, color = 'royalblue', label = 'clean')
            plt.plot(data.test_batch+batch_size, data.dirty_correct, color = 'wheat', label='_nolegend_')
            plt.scatter(data.test_batch+batch_size, data.dirty_correct, color = 'darksalmon', label = 'dirty')
            plt.xlabel('test batch number')
            plt.ylabel('0 = wrong, 1 = correct')
            plt.title('%s Data Trained: %s, Batch Size %d' %(data_name.title(), train_type.title(), batch_size))
            plt.yticks([0,1])
            plt.ylim([-.3, 1.3])
#             plt.xticks(range())
            plt.legend()
            plt.show()
            
def get_accuracy(analysis_df):
    a = analysis_df[analysis_df.test_batch.isnull() == False].groupby(['train_type', 'batch_size']).agg({'batch_size': 'count', 'clean_correct':'sum', 'dirty_correct':'sum'})
    a['accuracy'] = (a['clean_correct'] + a['dirty_correct'])/(2*a['batch_size'])
    return a[['accuracy']].reset_index()



