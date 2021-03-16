import numpy as np
import scipy.stats as stats

from models import Logit, MNLogit, PooledLinearModelFactory
from model_training.imputation import mice_impute_dataframe
from model_training.performance_metrics import get_model_performance

def perform_model_training(train_data_df, y, imputed_cols, M = 20, _type = 'log', get_pooled_parameters = False):
    imputed_train_dfs = []
    fitted_models = []
    
    for m in range(M):
        imputed_train_df = mice_impute_dataframe(train_data_df)

        X = imputed_train_df[imputed_cols].to_numpy()
        
        if _type == 'mnlog':
            estimator = MNLogit()
        else:
            estimator = Logit()
            
        estimator.fit(X, y)
        
        imputed_train_dfs.append(imputed_train_df)
        fitted_models.append(estimator)
        
    pooled_lin_model = PooledLinearModelFactory.init_from_linear_models(fitted_models, _type = _type)
    
    acc_measure_values = np.zeros(M)
    auc_values = np.zeros(M)
    citls = []
    cal_slopes = []
    prop_trues = []
    prop_preds = []
    
    for m in range(M):
        imputed_train_df = imputed_train_dfs[m]
        
        X = imputed_train_df[imputed_cols].to_numpy()
        
        acc_measure, auc, citl, cal_slope, prop_true, prop_pred = get_model_performance(pooled_lin_model, X, y, _type = _type)
        
        acc_measure_values[m] = acc_measure
        auc_values[m] = auc
        
        citls.append(citl)
        cal_slopes.append(cal_slope)
        
        prop_trues.append(prop_true)
        prop_preds.append(prop_pred)
       
    citls = np.array(citls)
    cal_slopes = np.array(cal_slopes)
    
    result_struct = {
        'acc_measure_values': acc_measure_values,
        'acc_measure': np.median(acc_measure_values),
        'acc_measure_iqr': stats.iqr(acc_measure_values),
        'auc_values': auc_values,
        'auc': np.median(auc_values),
        'auc_iqr': stats.iqr(auc_values),
        'citls': citls,
        'citl': np.median(citls, axis = 0),
        'citl_iqr': stats.iqr(citls, axis = 0),
        'cal_slopes': cal_slopes,
        'cal_slope': np.median(cal_slopes, axis = 0),
        'cal_slope_iqr': stats.iqr(cal_slopes, axis = 0),
        'prop_trues': prop_trues,
        'prop_preds': prop_preds
    }
    
    if get_pooled_parameters:
        pooled_model_parameters = _pool_model_parameters(fitted_models, train_data_df.shape[0], imputed_cols)
        
        return result_struct, pooled_lin_model, imputed_train_dfs, pooled_model_parameters
    else:
        return result_struct, pooled_lin_model, imputed_train_dfs

def _pool_model_parameters(fitted_models, N, imputed_cols):
    k = len(imputed_cols) + 1
    dfcom = N - k
    
    no_models = len(fitted_models)
    param_shape = fitted_models[0].fitted_model.params.shape
    D = param_shape[0]
    if len(param_shape) == 1:
        n_outcomes = 1
    else:
        n_outcomes = param_shape[1]
        
    param_list = imputed_cols.copy()
    param_list.insert(0, 'BIAS')
    pooled_parameters = {
        'columns': param_list
    }
    
    for outcome in range(n_outcomes):
        pooled_coefs = np.zeros(D)
        pooled_ses = np.zeros(D)
        pooled_p_vals = np.zeros(D)
    
        coefs = np.zeros((no_models, D))
        variances = np.zeros((no_models, D))
    
        cov_start_idx = 0 + outcome * D
        cov_end_idx = D + outcome * D
        for n, fitted_model in enumerate(fitted_models):
            params = fitted_model.fitted_model.params
            if params.ndim == 1:
                params = np.reshape(params, (-1, 1))
            
            coefs[n, :] = params[:, outcome]
            variances[n, :] = np.diag(fitted_model.fitted_model.cov_params())[cov_start_idx:cov_end_idx]
        
        for idx, d in enumerate(range(D)):
            _coefs = coefs[:, d]
            _variances = variances[:, d]

            ubar = np.mean(_variances)
            b = np.var(_coefs, ddof = 1)

            t = ubar + (1 + 1 / N) * b

            pooled_coef = np.mean(_coefs)
            pooled_se = np.sqrt(t)

            _lambda = (1 + 1 / N) * b / t
            _lambda = np.maximum(_lambda, 1e-4)
            dfold =  (N - 1) / np.power(_lambda, 2)
            dfobs = (dfcom + 1) / (dfcom + 3) * dfcom * (1 - _lambda)

            dfcom = dfold * dfobs / (dfold + dfobs)

            pooled_coefs[idx] = pooled_coef
            pooled_ses[idx] = pooled_se
            pooled_p_vals[idx] = stats.t.sf(np.absolute(pooled_coef / pooled_se), dfcom, loc=0, scale=1) * 2
            
        outcome_label = outcome + 1
        pooled_parameters[f'pooled_coefs_{outcome_label}'] = pooled_coefs.tolist()
        pooled_parameters[f'pooled_ses_{outcome_label}'] = pooled_ses.tolist()
        pooled_parameters[f'pooled_p_vals_{outcome_label}'] = pooled_p_vals.tolist()
        
    return pooled_parameters