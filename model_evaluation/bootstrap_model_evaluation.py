import logging
import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import roc_auc_score

from model_training import perform_model_training
from utils import create_result_directories, init_logging, init_results_struct

def perform_bootstrap_model_evaluation(das_type, treatment, bootstrap_idx, outcome_col = 'class_bin', M = 20, _type = 'log'):
    init_logging()
    
    logging.info(f'perform_bootstrap_model_evaluation: das_type: {das_type}, treatment: {treatment}, outcome_col: {outcome_col}, boot_idx: {bootstrap_idx}, _type: {_type}')
    
    train_data_df = pd.read_csv(f'./data/das28_BIOP_{das_type}_{treatment}_outcomes.csv')
    # Remove samples with missing outcome - only relevant for 2c outcomes
    train_data_df = train_data_df.iloc[np.where(~pd.isnull(train_data_df[outcome_col]))[0]].reset_index(drop = True)
    data_cols = ['eular_bin', 'das_tend.0', 'das_vas.0', 'das_swol.0', f'{das_type}.0', 'FIRSTBIO', 'WEIGHT', 'HEIGHT', 'DISDUR', 'AGEONSET', 'HAQ', 'SEX', 'SERO', 'CONCURRENT_DMARD', 'HAD_D', 'HAD_A']
    imputed_cols = ['das_tend.0', 'das_vas.0', 'das_swol.0', f'{das_type}.0', 'FIRSTBIO', 'WEIGHT', 'HEIGHT', 'DISDUR', 'AGEONSET', 'HAQ', 'SEX', 'SERO', 'CONCURRENT_DMARD', 'HAD_D', 'HAD_A']
    
    y = train_data_df[outcome_col].to_numpy()
    
    train_data_df = train_data_df[data_cols]
    
    N = train_data_df.shape[0]
    pop = np.arange(N)
    
    imputed_train_dfs = []
    for m in range(M):
        imputed_train_df = pd.read_csv(f'./data/results/{treatment}/{outcome_col}/imputed/das28_{das_type}_{treatment}_{outcome_col}_outcomes_mice_imputed_{m}.csv')
        imputed_train_dfs.append(imputed_train_df)
        
    logging.info('Drawing Bootstrap Sample')
    is_valid_bootstrap = False
    while(not is_valid_bootstrap):
        boot_idx = np.random.choice(pop, N)
        boot_train_data_df = train_data_df.iloc[boot_idx].reset_index(drop = True)
        boot_y = y[boot_idx]
        
        is_valid_bootstrap = _is_valid_bootstrap(boot_train_data_df, y, _type)
        
    logging.info('Begin Training')
        
    boot_result_struct, boot_pooled_lin_model, boot_imputed_train_dfs = perform_model_training(boot_train_data_df, boot_y, imputed_cols, _type = _type)
    results = init_results_struct(_type, M)
    
    for m in range(M):
        imputed_train_df = imputed_train_dfs[m]
            
        X = imputed_train_df[imputed_cols].to_numpy()
        
        results.add_result(m, boot_pooled_lin_model, X, y)

    result_struct = results.to_results_struct()
    
    result_struct['boot_apparent_performance'] = boot_result_struct
    
    logging.info(result_struct)
    
    pickle.dump(result_struct, open(f"./data/results/{treatment}/{outcome_col}/bootstraps/{das_type}_{treatment}_{outcome_col}_bootstrap_{bootstrap_idx}_eval.data", "wb"))
    
def _is_valid_bootstrap(boot_train_data_df, y, _type):
    is_valid_bootstrap = True
    
    y_unique = np.unique(y)
    if _type == 'log':
        if len(y_unique) != 2:
            logging.info('Invalid values for binary outcome y')
            is_valid_bootstrap = False
    elif _type == 'mnlog':
        if len(y_unique) != 3:
            logging.info('Invalid values for multinomial outcome y')
            is_valid_bootstrap = False
    
    #not_null_idx = np.where(~pd.isnull(boot_train_data_df['SMOKE']))
    #n_smoke_cats = len(np.unique(boot_train_data_df.iloc[not_null_idx]['SMOKE']))
    #if n_smoke_cats != 3:
        #logging.info(f'Invalid values for mn category SMOKE')
        
        #is_valid_bootstrap = False
    else:
        for bin_cat in ['FIRSTBIO', 'SEX', 'SERO', 'CONCURRENT_DMARD']:
            not_null_idx = np.where(~pd.isnull(boot_train_data_df[bin_cat]))
            n_cats = len(np.unique(boot_train_data_df.iloc[not_null_idx][bin_cat]))

            if n_cats != 2:
                logging.info(f'Invalid values for bin category {bin_cat}')
                
                is_valid_bootstrap = False
                break
    if not is_valid_bootstrap:
        logging.info('Invalid bootstrap, drawing anew')
        
    return is_valid_bootstrap