import logging
import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import roc_auc_score

from model_training import get_model_performance, perform_model_training
from utils import create_result_directories, init_logging

def perform_bootstrap_model_evaluation(das_type, treatment, bootstrap_idx, outcome_col = 'class_bin', M = 20, _type = 'log'):
    init_logging()
    
    logging.info(f'perform_bootstrap_model_evaluation: das_type: {das_type}, treatment: {treatment}, outcome_col: {outcome_col}, boot_idx: {bootstrap_idx}, _type: {_type}')
    
    train_data_df = pd.read_csv(f'./data/das28_BIOP_{das_type}_{treatment}_outcomes.csv')
    data_cols = ['eular_bin', 'das_tend.0', 'das_vas.0', 'das_swol.0', f'{das_type}.0', 'FIRSTBIO', 'WEIGHT', 'HEIGHT', 'DISDUR', 'SMOKE', 'AGEONSET', 'HAQ', 'SEX', 'SERO', 'CONCURRENT_DMARD']
    imputed_cols = ['das_tend.0', 'das_vas.0', 'das_swol.0', f'{das_type}.0', 'FIRSTBIO', 'WEIGHT', 'HEIGHT', 'DISDUR', 'SMOKE_current', 'SMOKE_past', 'AGEONSET', 'HAQ', 'SEX', 'SERO', 'CONCURRENT_DMARD']
    
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
        
    raw_acc_measure_values = np.zeros(M)
    raw_auc_values = np.zeros(M)
    raw_citls = []
    raw_cal_slopes = []
    raw_prop_trues = []
    raw_prop_preds = []
    
    for m in range(M):
        imputed_train_df = imputed_train_dfs[m]
            
        X = imputed_train_df[imputed_cols].to_numpy()
        
        acc_measure, auc, citl, cal_slope, prop_true, prop_pred = get_model_performance(boot_pooled_lin_model, X, y, _type = _type)
            
        raw_acc_measure_values[m] = acc_measure
        raw_auc_values[m] = auc
        raw_citls.append(citl)
        raw_cal_slopes.append(cal_slope)
        raw_prop_trues.append(prop_true)
        raw_prop_preds.append(prop_pred)
       
    raw_citls = np.array(raw_citls)
    raw_cal_slopes = np.array(raw_cal_slopes)
    
    raw_acc_measure = np.median(raw_acc_measure_values)
    raw_auc = np.median(raw_auc_values)
    raw_citl = np.median(raw_citls, axis = 0)
    raw_cal_slope = np.median(raw_cal_slopes, axis = 0)
        
    result_struct = {
        'boot_apparent_performance': boot_result_struct,
        'raw_acc_measure_values': raw_acc_measure_values,
        'raw_acc_measure': raw_acc_measure,
        'acc_measure_optimism': boot_result_struct['acc_measure'] - raw_acc_measure,
        'raw_auc_values': raw_auc_values,
        'raw_auc': raw_auc,
        'auc_optimism': boot_result_struct['auc'] - raw_auc,
        'raw_citls': raw_citls,
        'raw_citl': raw_citl,
        'citl_optimism': boot_result_struct['citl'] - raw_citl,
        'raw_cal_slopes': raw_cal_slopes,
        'raw_cal_slope': raw_cal_slope,
        'cal_slope_optimism': boot_result_struct['cal_slope'] - raw_cal_slope,
        'raw_prop_trues': raw_prop_trues,
        'raw_prop_preds': raw_prop_preds,
    }
    
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
    
    n_smoke_cats = len(np.unique(boot_train_data_df['SMOKE']))
    if n_smoke_cats != 3:
        logging.info(f'Invalid values for mn category SMOKE')
        
        is_valid_bootstrap = False
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