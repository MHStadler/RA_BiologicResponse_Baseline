import numpy as np

from models import Logit
from utils import calibration_curve

from sklearn.metrics import accuracy_score, brier_score_loss, mean_squared_error, roc_auc_score, r2_score

def get_model_performance(pooled_lin_model, X, y, _type = 'log'):
    y_prob = pooled_lin_model.predict_proba(X)
    lin_pred = pooled_lin_model.lin_pred(X)
    
    if _type == 'cont':
        acc_measure = mean_squared_error(y, y_prob)
        r2 = r2_score(y, y_prob)
        
        return acc_measure, r2
    else:
        if _type == 'log':
            auc = roc_auc_score(y, y_prob)

            acc_measure = brier_score_loss(y, y_prob)
            citl, cal_slope = _get_bin_calibration(lin_pred, y)
            prop_true, prop_pred = calibration_curve(y, y_prob, n_bins = 10, strategy = 'quantile')
        elif _type == 'mnlog':
            auc = roc_auc_score(y, y_prob, average = 'macro', multi_class = 'ovo')

            acc_measure = accuracy_score(y, np.argmax(y_prob, axis = 1))
            citl, cal_slope = _get_mn_calibration(lin_pred, y)

            prop_true_one, prop_pred_one = calibration_curve((y == 0).astype(int), y_prob[:, 0], n_bins = 10, strategy = 'quantile')
            prop_true_two, prop_pred_two = calibration_curve((y == 1).astype(int), y_prob[:, 1], n_bins = 10, strategy = 'quantile')
            prop_true_three, prop_pred_three = calibration_curve((y == 2).astype(int), y_prob[:, 2], n_bins = 10, strategy = 'quantile')

            prop_true = [prop_true_one, prop_true_two, prop_true_three]
            prop_pred = [prop_pred_one, prop_pred_two, prop_pred_three]
            
            return acc_measure, auc, citl, cal_slope, prop_true, prop_pred

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
    
    return [citl_one, citl_two], [cal_slope_one, cal_slope_two]