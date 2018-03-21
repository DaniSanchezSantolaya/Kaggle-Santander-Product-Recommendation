# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 11:15:39 2018

@author: santd1
The script preprocess the data from Santander csv dataset.
It creates a pickle file, which contains a Train and Test set.
Train set is used creating data samples until the specified last_date_train.
We create a sample for each month that the user adds a product to create more
samples (data augmentation)
The test set contains the labels that of the month specified in date_test, while
the features are constructed using previous months data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy import sparse
import pickle
import gc


# Preprocessing parameters
max_seq_length = 18
num_inputs = 48
num_outputs = 24
last_date_train = '2016-04-28'
date_test = '2016-05-28'
name_dataset = 'april_2016'


# p_train = 0.8 # Proportion of train samples - rest users will go to validation



# Product columns
product_columns = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1',
                  'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
                  'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
                  'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1',
                  'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
                  'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
   
  
# Read data
print('Read data...')
df_train = pd.read_csv('train_ver2.csv', usecols = product_columns + ['fecha_dato', 'ncodpers'])
df_test = pd.read_csv('test_ver2.csv')

df_train.ind_nomina_ult1.fillna(0, inplace=True)
df_train.ind_nom_pens_ult1.fillna(0, inplace=True)

# Transform type product features to save memory
for f in product_columns:
    df_train[f] = df_train[f].astype(np.int8)    

df_train['fecha_dato'] = pd.to_datetime(df_train['fecha_dato'])


gc.collect()

dataset = {}
dataset['train'] = {}
dataset['train']['X'] = []
dataset['train']['Y'] = []
dataset['train']['ncodpers'] = []
dataset['train']['label_distribution'] = []
dataset['test'] = {}
dataset['test']['X'] = []
dataset['test']['Y'] = []
dataset['test']['ncodpers'] = []
dataset['test']['label_distribution'] = []


n = 0
print('Start preprocessing')
for ncodpers,group in df_train.groupby('ncodpers'):
    #group_sorted = group.sort_values('fecha_dato')
    group_sorted_train = group[group.fecha_dato <= last_date_train].sort_values('fecha_dato')
    interactions = group_sorted_train[product_columns].astype(np.int32).diff().fillna(0).values
    interactions_mask = np.sum(np.abs(interactions), axis = 1) > 0
    interactions = interactions[interactions_mask > 0, :] # Remove if we want to include the times
    nInteractions = np.sum(np.abs(interactions))
    user_interaction_list = []
    
    # Train 
    if nInteractions > 0:
        if len(interactions) > 1:
            if len(np.where(interactions[1:] == 1)[0]) > 0: # We look that there is some positive interaction after the first one, otherwise we don't have anything to learn
                idx_pos_interactions = []
                for t in range(len(interactions)):
                    # temporal features (interactions)
                    added = np.where(interactions[t] == 1)[0]
                    removed = np.where(interactions[t] == -1)[0] + 24
                    interactions_joined = list(added) + list(removed)
                    if len(interactions_joined) > 0:
                        user_interaction_list.append(interactions_joined)
                        # user_interaction_list.append((t, interactions_joined)) Use this if we want to include the times
                    if len(added) > 0:
                        idx_pos_interactions.append(len(user_interaction_list) - 1)

                for idx_pos in idx_pos_interactions:
                    x = np.zeros((max_seq_length, num_inputs), np.int8)
                    y = np.zeros((num_outputs), np.int8)
                    # if it's the first interaction we don't create a training sample, as there is no past interaction data
                    if idx_pos > 0:
                        # FEATURES
                        for i in range(idx_pos):
                            x[i, user_interaction_list[i]] = 1
                        # LABELS
                        filter_pos_int = np.array(user_interaction_list[idx_pos]) < 24
                        y[np.array(user_interaction_list[idx_pos])[filter_pos_int]] = 1
                        # Add to the correspondent set
                        dataset['train']['X'].append(sparse.csr_matrix(x, dtype=np.int8))
                        dataset['train']['Y'].append(sparse.csr_matrix(y, dtype=np.int8))
                        dataset['train']['ncodpers'].append(ncodpers)
                        dataset['train']['label_distribution'].extend(list(np.array(user_interaction_list[idx_pos])[filter_pos_int]))

                
    # Test
    if nInteractions > 0:
        if len(interactions) > 1:
            x = np.zeros((max_seq_length, num_inputs), np.int8)
            y = np.zeros((num_outputs), np.int8)
            # if the user interaction list it has not been obtained by creating the train sample, we create it
            if len(user_interaction_list) == 0:
                for t in range(len(interactions)):
                    # temporal features (interactions)
                    added = np.where(interactions[t] == 1)[0]
                    removed = np.where(interactions[t] == -1)[0] + 24
                    interactions_joined = list(added) + list(removed)
                    if len(interactions_joined) > 0:
                        user_interaction_list.append(interactions_joined)
                        # user_interaction_list.append((t, interactions_joined)) Use this if we want to include the times
            # Features
            for i in range(len(user_interaction_list)):
                x[i, user_interaction_list[i]] = 1
            # Labels
            interactions_test_date = group[group.fecha_dato <= date_test][product_columns].astype(np.int32).diff().fillna(0).values
            interactions_test_date = interactions_test_date[-1]
            num_adds_test_date = np.sum(interactions_test_date == 1)
            # Add sample only if there is some positive interaction
            if num_adds_test_date > 0:
                interaction_idx = np.where(interactions_test_date == 1)[0]
                y[interaction_idx] = 1
                dataset['val']['X'].append(sparse.csr_matrix(x, dtype=np.int8))
                dataset['val']['Y'].append(sparse.csr_matrix(y, dtype=np.int8))
                dataset['val']['ncodpers'].append(ncodpers)
                dataset['val']['label_distribution'].extend(list(interaction_idx))
                

        
    if n % 50000 == 0:
        print(n)
    n += 1
    

print('Num train samples: ' + str(len(dataset['train']['X'])))
print('Num test samples: ' + str(len(dataset['test']['X'])))

with open("preprocessed/dataset_augmented_" + last_date_train + ".pickle", 'wb') as handle:
    pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
