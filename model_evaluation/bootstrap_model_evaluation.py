import logging
import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import roc_auc_score

from model_training import get_model_performance, perform_model_training
from utils import init_logging

def perform_bootstrap_model_evaluation(das_type, bootstrap_idx, M = 20):
    init_logging()
    
    logging.info(f'perform_bootstrap_model_evaluation: das_type: {das_type}, boot_idx: {bootstrap_idx}')
    
    train_data_df = pd.read_csv(f'./data/das28_BIOP_{das_type}_outcomes.csv')
    data_cols = ['das_tend.0', 'das_vas.0', 'das_swol.0', f'{das_type}.0', 'FIRSTBIO', 'WEIGHT', 'HEIGHT', 'DISDUR', 'SMOKE', 'BIO', 'AGEONSET', 'HAQ', 'SEX', 'SERO', 'CONCURRENT_DMARD']
    imputed_cols = ['das_tend.0', 'das_vas.0', 'das_swol.0', f'{das_type}.0', 'FIRSTBIO', 'WEIGHT', 'HEIGHT', 'DISDUR', 'SMOKE_current', 'SMOKE_past', 'BIO_adalimumab', 'BIO_infliximab', 'AGEONSET', 'HAQ', 'SEX', 'SERO', 'CONCURRENT_DMARD']
    
    y = train_data_df['class'] < 1
    y = y.astype(int).to_numpy()
    
    train_data_df = train_data_df[data_cols]
    
    N = train_data_df.shape[0]
    pop = np.arange(N)
    
    imputed_train_dfs = []
    for m in range(M):
        imputed_train_df = pd.read_csv(f'./data/imputed/das28_{das_type}_outcomes_mice_imputed_{m}.csv')
        imputed_train_dfs.append(imputed_train_df)
    
    boot_idx = np.random.choice(pop, N)
        
    boot_train_data_df = train_data_df.iloc[boot_idx].reset_index(drop = True)
    boot_y = y[boot_idx]
        
    boot_result_struct, boot_pooled_lin_model, boot_pooled_scaler, boot_imputed_train_dfs = perform_model_training(boot_train_data_df, boot_y, imputed_cols)
        
    raw_brier_score_values = np.zeros(M)
    raw_auc_values = np.zeros(M)
    raw_citls = np.zeros(M)
    raw_cal_slopes = np.zeros(M)
    
    for m in range(M):
        imputed_train_df = imputed_train_dfs[m]
            
        X = imputed_train_df[imputed_cols].to_numpy()
        X = boot_pooled_scaler.transform(X)
        
        brier_score, auc, citl, cal_slope = get_model_performance(boot_pooled_lin_model, X, y)
            
        raw_brier_score_values[m] = brier_score
        raw_auc_values[m] = auc
        raw_citls[m] = citl
        raw_cal_slopes[m] = cal_slope
       
    raw_brier_score = np.mean(raw_brier_score_values)
    raw_auc = np.median(raw_auc_values)
    raw_citl = np.mean(raw_citls)
    raw_cal_slope = np.mean(raw_cal_slopes)
        
    result_struct = {
        'boot_apparent_performance': boot_result_struct,
        'raw_brier_score_values': raw_brier_score_values,
        'raw_brier_score': raw_brier_score,
        'brier_score_optimism': boot_result_struct['brier_score'] - raw_brier_score,
        'raw_auc_values': raw_auc_values,
        'raw_auc': raw_auc,
        'auc_optimism': boot_result_struct['auc'] - raw_auc,
        'raw_citls': raw_citls,
        'raw_citl': raw_citl,
        'citl_optimism': boot_result_struct['citl'] - raw_citl,
        'raw_cal_slopes': raw_cal_slopes,
        'raw_cal_slope': raw_cal_slope,
        'cal_slope_optimism': boot_result_struct['cal_slope'] - raw_cal_slope
    }
    
    logging.info(result_struct)
    
    pickle.dump(result_struct, open(f"./data/results/{das_type}/bootstraps/{das_type}_bootstrap_{bootstrap_idx}_eval.data", "wb"))