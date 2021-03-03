import json
import logging
import pandas as pd
import pickle

from model_training import perform_model_training
from utils import init_logging

def perform_apparent_model_evaluation(das_type, M = 20):
    init_logging()
    
    
    logging.info(f'perform_apparent_model_evaluation: das_type: {das_type}')
    
    train_data_df = pd.read_csv(f'./data/das28_BIOP_{das_type}_outcomes.csv')
    
    data_cols = ['das_tend.0', 'das_vas.0', 'das_swol.0', f'{das_type}.0', 'FIRSTBIO', 'WEIGHT', 'HEIGHT', 'DISDUR', 'SMOKE', 'BIO', 'AGEONSET', 'HAQ', 'SEX', 'SERO', 'CONCURRENT_DMARD']
    imputed_cols = ['das_tend.0', 'das_vas.0', 'das_swol.0', f'{das_type}.0', 'FIRSTBIO', 'WEIGHT', 'HEIGHT', 'DISDUR', 'SMOKE_current', 'SMOKE_past', 'BIO_adalimumab', 'BIO_infliximab', 'AGEONSET', 'HAQ', 'SEX', 'SERO', 'CONCURRENT_DMARD']

    y = train_data_df['class'] < 1
    y = y.astype(int).to_numpy()
    
    train_data_df = train_data_df[data_cols]
    
    result_struct, pooled_lin_model, pooled_scaler, imputed_train_dfs, pooled_parameters = perform_model_training(train_data_df, y, imputed_cols, get_pooled_parameters = True)
    
    for idx, imputed_train_df in enumerate(imputed_train_dfs):
        imputed_train_df.to_csv(f'./data/imputed/das28_{das_type}_outcomes_mice_imputed_{idx}.csv', index = False)
        
    pooled_lin_model.to_file(f'{das_type}_pooled_model')
    pooled_scaler.to_file(f'{das_type}_pooled_scaler')
       
    logging.info(result_struct)
    
    pickle.dump(result_struct, open(f"./data/results/{das_type}/{das_type}_apparent_eval.data", "wb"))
    
    with open(f'./data/results/{das_type}/pooled_model_parameters.json', 'w') as outfile:
        json.dump(pooled_parameters, outfile)