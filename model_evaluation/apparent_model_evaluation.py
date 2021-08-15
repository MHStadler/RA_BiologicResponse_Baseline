import json
import logging
import numpy as np
import pandas as pd
import pickle

from model_training import perform_model_training
from utils import create_result_directories, init_logging

def perform_apparent_model_evaluation(das_type, treatment, outcome_col = 'class_bin', M = 20, _type = 'log'):
    init_logging()
    
    logging.info(f'perform_apparent_model_evaluation: das_type: {das_type} - treatment: {treatment} - outcome_col: {outcome_col} - _type: {_type}')
    
    create_result_directories(das_type, treatment, outcome_col = outcome_col)
    
    train_data_df = pd.read_csv(f'./data/das28_BIOP_{das_type}_{treatment}_outcomes.csv')
    # Remove samples with missing outcome - only relevant for 2c outcomes
    train_data_df = train_data_df.iloc[np.where(~pd.isnull(train_data_df[outcome_col]))[0]].reset_index(drop = True)
    
    data_cols = ['eular_bin', 'das_tend.0', 'das_vas.0', 'das_swol.0', f'{das_type}.0', 'FIRSTBIO', 'WEIGHT', 'HEIGHT', 'DISDUR', 'AGEONSET', 'HAQ', 'SEX', 'SERO', 'CONCURRENT_DMARD', 'HAD_D', 'HAD_A']
    imputed_cols = ['das_tend.0', 'das_vas.0', 'das_swol.0', f'{das_type}.0', 'FIRSTBIO', 'WEIGHT', 'HEIGHT', 'DISDUR', 'AGEONSET', 'HAQ', 'SEX', 'SERO', 'CONCURRENT_DMARD', 'HAD_D', 'HAD_A']

    y = train_data_df[outcome_col].to_numpy()
    
    train_data_df = train_data_df[data_cols]
    
    result_struct, pooled_lin_model, imputed_train_dfs, pooled_parameters = perform_model_training(train_data_df, y, imputed_cols, get_pooled_parameters = True, _type = _type)
    
    for idx, imputed_train_df in enumerate(imputed_train_dfs):
        imputed_train_df.to_csv(f'./data/results/{treatment}/{outcome_col}/imputed/das28_{das_type}_{treatment}_{outcome_col}_outcomes_mice_imputed_{idx}.csv', index = False)
        
    pooled_lin_model.to_file(f'./data/results/{treatment}/{outcome_col}/{das_type}_{treatment}_{outcome_col}_pooled_model')
       
    logging.info(result_struct)
    
    pickle.dump(result_struct, open(f"./data/results/{treatment}/{outcome_col}/{das_type}_{treatment}_{outcome_col}_apparent_eval.data", "wb"))
    
    with open(f'./data/results/{treatment}/{outcome_col}/{das_type}_{treatment}_{outcome_col}_pooled_model_parameters.json', 'w') as outfile:
        json.dump(pooled_parameters, outfile)