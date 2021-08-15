import logging
import numpy as np
import os
import scipy.stats as stats

from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score, r2_score

from models import Logit
from utils.scikit import calibration_curve

def create_result_directories(das_type, treatment, outcome_col = 'class_bin'):
    try:
        os.makedirs(f'./data/results/{treatment}/{outcome_col}/imputed', exist_ok = True)
        os.makedirs(f'./data/results/{treatment}/{outcome_col}/bootstraps', exist_ok = True)
    except OSError:
        logging.error('Failed to created output directories')
    else:
        logging.info('Created output directories')

def init_results_struct(_type, M):
    if _type == 'log':
        return LogResultsStruct(M)
    elif _type == 'cont':
        return LinearResultsStruct(M)
        
class LogResultsStruct():
    def __init__(self, M):
        self.M = M
        
        self.brier_score_values = np.zeros(M)
        self.auc_values = np.zeros(M)
        self.citl_values = np.zeros(M)
        self.cal_slope_values = np.zeros(M)
        
        self.prop_trues = []
        self.prop_preds = []
        
    def add_result(self, m, model, X, y):
        y_prob = model.predict_proba(X)
        lin_pred = model.lin_pred(X)
        
        self.brier_score_values[m] = brier_score_loss(y, y_prob)
        self.auc_values[m] = roc_auc_score(y, y_prob)
        
        citl, cal_slope = _get_bin_calibration(lin_pred, y)
        self.citl_values[m] = citl
        self.cal_slope_values[m] = cal_slope
        
        prop_true, prop_pred = calibration_curve(y, y_prob, n_bins = 10, strategy = 'quantile')
        
        self.prop_trues.append(prop_true)
        self.prop_preds.append(prop_pred)
        
    def to_results_struct(self):
        result_struct = {
            'brier_score_values': self.brier_score_values,
            'brier_score': np.median(self.brier_score_values),
            'brier_score_iqr': stats.iqr(self.brier_score_values),
            'auc_values': self.auc_values,
            'auc': np.median(self.auc_values),
            'auc_iqr': stats.iqr(self.auc_values),
            'citls': self.citl_values,
            'citl': np.median(self.citl_values, axis = 0),
            'citl_iqr': stats.iqr(self.citl_values, axis = 0),
            'cal_slopes': self.cal_slope_values,
            'cal_slope': np.median(self.cal_slope_values, axis = 0),
            'cal_slope_iqr': stats.iqr(self.cal_slope_values, axis = 0),
            'prop_trues': self.prop_trues,
            'prop_preds': self.prop_preds
        }
        
        return result_struct
    
class LinearResultsStruct():
    def __init__(self, M):
        self.M = M
        
        self.mse_values = np.zeros(M)
        self.mae_values = np.zeros(M)
        self.r2_values = np.zeros(M)
        
    def add_result(self, m, model, X, y):
        y_pred = model.predict(X)
        
        self.mse_values[m] = mean_squared_error(y, y_pred)
        self.mae_values[m] = mean_absolute_error(y, y_pred)
        self.r2_values[m] = r2_score(y, y_pred)
        
    def to_results_struct(self):
        result_struct = {
            'mse_values': self.mse_values,
            'mse': np.median(self.mse_values),
            'mse_values_iqr': stats.iqr(self.mse_values),
            'mae_values': self.mae_values,
            'mae': np.median(self.mae_values),
            'mae_values_iqr': stats.iqr(self.mae_values),
            'r2_values': self.r2_values,
            'r2': np.median(self.r2_values),
            'r2_iqr': stats.iqr(self.r2_values)
        }
        
        return result_struct
        
def _get_bin_calibration(lin_pred, y):
    cal_model = Logit()
    cal_model.fit(lin_pred, y)
    citl = cal_model.fitted_model.params[0]
    cal_slope = cal_model.fitted_model.params[1]
    
    return citl, cal_slope