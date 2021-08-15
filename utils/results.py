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
    elif _type == 'mnlog':
        return MNLogResultsStruct(M)
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
    
class MNLogResultsStruct():
    def __init__(self, M):
        self.M = M
        
        self.accuracy_values = np.zeros(M)
        self.auc_values = np.zeros(M)
        self.citl_values_ones = np.zeros(M)
        self.cal_slope_values_ones = np.zeros(M)
        self.citl_values_twos = np.zeros(M)
        self.cal_slope_values_twos = np.zeros(M)
        
        self.prop_trues_0 = []
        self.prop_preds_0 = []
        self.prop_trues_1 = []
        self.prop_preds_1 = []
        self.prop_trues_2 = []
        self.prop_preds_2 = []
        
    def add_result(self, m, model, X, y):
        y_prob = model.predict_proba(X)
        lin_pred = model.lin_pred(X)
        
        self.accuracy_values[m] = accuracy_score(y, np.argmax(y_prob, axis = 1))
        self.auc_values[m] = roc_auc_score(y, y_prob, average = 'macro', multi_class = 'ovo')
        
        citl_one, cal_slope_one, citl_two, cal_slope_two = _get_mn_calibration(lin_pred, y)
        
        self.citl_values_one[m] = citl_one
        self.cal_slope_values_one[m] = cal_slope_one
        self.citl_values_two[m] = citl_two
        self.cal_slope_values_two[m] = cal_slope_two
        
        prop_true_one, prop_pred_one = calibration_curve((y == 0).astype(int), y_prob[:, 0], n_bins = 10, strategy = 'quantile')
        prop_true_two, prop_pred_two = calibration_curve((y == 1).astype(int), y_prob[:, 1], n_bins = 10, strategy = 'quantile')
        prop_true_three, prop_pred_three = calibration_curve((y == 2).astype(int), y_prob[:, 2], n_bins = 10, strategy = 'quantile')
        
        self.prop_trues_0.append(prop_true_one)
        self.prop_preds_0.append(prop_pred_one)
        self.prop_trues_1.append(prop_true_two)
        self.prop_preds_1.append(prop_pred_two)
        self.prop_trues_2.append(prop_true_three)
        self.prop_preds_2.append(prop_pred_three)
        
    def to_results_struct(self):
        result_struct = {
            'accuracy_values': self.accuracy_values,
            'accuracy': np.median(self.accuracy_values),
            'accuracy_iqr': stats.iqr(self.accuracy_values),
            'auc_values': self.auc_values,
            'auc': np.median(self.auc_values),
            'auc_iqr': stats.iqr(self.auc_values),
            'citls_1': self.citl_values_one,
            'citl_1': np.median(self.citl_values_one, axis = 0),
            'citl_1_iqr': stats.iqr(self.citl_values_one, axis = 0),
            'cal_slopes_1': self.cal_slope_values_one,
            'cal_slope_1': np.median(self.cal_slope_values_one, axis = 0),
            'cal_slope_1_iqr': stats.iqr(self.cal_slope_values_one, axis = 0),
            'citls_2': self.citl_values_two,
            'citl_2': np.median(self.citl_values_two, axis = 0),
            'citl_2_iqr': stats.iqr(self.citl_values_two, axis = 0),
            'cal_slopes_2': self.cal_slope_values_two,
            'cal_slope_2': np.median(self.cal_slope_values_two, axis = 0),
            'cal_slope_2_iqr': stats.iqr(self.cal_slope_values_two, axis = 0),
            'prop_trues_0': self.prop_trues_0,
            'prop_preds_0': self.prop_preds_0,
            'prop_trues_1': self.prop_trues_1,
            'prop_preds_1': self.prop_preds_1,
            'prop_trues_2': self.prop_trues_2,
            'prop_preds_2': self.prop_preds_2
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

def _get_mn_calibration(lin_pred, y):
    one_lin_pred = lin_pred[:, 0]
    two_lin_pred = lin_pred[:, 1]
    
    y_one = (y == 1).astype(int)
    y_two = (y == 2).astype(int)

    cal_model_one = Logit()
    cal_model_one.fit(one_lin_pred, y_one)
    citl_one = cal_model_one.fitted_model.params[0]
    cal_slope_one = cal_model_one.fitted_model.params[1]
    
    cal_model_two = Logit()
    cal_model_two.fit(two_lin_pred, y_two)
    citl_two = cal_model_two.fitted_model.params[0]
    cal_slope_two = cal_model_two.fitted_model.params[1]
    
    return citl_one, cal_slope_one, citl_two, cal_slope_two