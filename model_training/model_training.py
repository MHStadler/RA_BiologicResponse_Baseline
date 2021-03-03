import numpy as np
import scipy.stats as stats

from sklearn.preprocessing import StandardScaler

from models import Logit, PooledLinearModel, PooledScaler
from model_training.imputation import mice_impute_dataframe
from model_training.performance_metrics import get_model_performance

def perform_model_training(train_data_df, y, imputed_cols, M = 20, get_pooled_parameters = False):
    imputed_train_dfs = []
    fitted_models = []
    scalers = []
    
    for m in range(M):
        imputed_train_df = mice_impute_dataframe(train_data_df)

        X = imputed_train_df[imputed_cols].to_numpy()
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        estimator = Logit()
        estimator.fit(X, y)
        
        imputed_train_dfs.append(imputed_train_df)
        fitted_models.append(estimator)
        scalers.append(scaler)
        
    pooled_lin_model = PooledLinearModel.init_from_linear_models(fitted_models)
    pooled_scaler = PooledScaler.init_from_scalers(scalers)
    
    brier_score_values = np.zeros(M)
    auc_values = np.zeros(M)
    citls = np.zeros(M)
    cal_slopes = np.zeros(M)
    
    for m in range(M):
        imputed_train_df = imputed_train_dfs[m]
        
        X = imputed_train_df[imputed_cols].to_numpy()
        X = scaler.transform(X)
        
        brier_score, auc, citl, cal_slope = get_model_performance(pooled_lin_model, X, y)
        
        brier_score_values[m] = brier_score
        auc_values[m] = auc
        citls[m] = citl
        cal_slopes[m] = cal_slope
        
    result_struct = {
        'brier_score_values': brier_score_values,
        'brier_score': np.median(brier_score_values),
        'brier_score_iqr': stats.iqr(brier_score_values),
        'auc_values': auc_values,
        'auc': np.median(auc_values),
        'auc_iqr': stats.iqr(auc_values),
        'citls': citls,
        'citl': np.median(citls),
        'citl_iqr': stats.iqr(citls),
        'cal_slopes': cal_slopes,
        'cal_slope': np.median(cal_slopes),
        'cal_slope_iqr': stats.iqr(cal_slopes)
    }
    
    if get_pooled_parameters:
        pooled_model_parameters = _pool_model_parameters(fitted_models, train_data_df.shape[0], imputed_cols)
        
        return result_struct, pooled_lin_model, pooled_scaler, imputed_train_dfs, pooled_model_parameters
    else:
        return result_struct, pooled_lin_model, pooled_scaler, imputed_train_dfs

def _pool_model_parameters(fitted_models, N, imputed_cols):
    k = len(imputed_cols) + 1
    dfcom = N - k
    
    no_models = len(fitted_models)
    D = fitted_models[0].fitted_model.params.shape[0]
    
    pooled_coefs = np.zeros(D)
    pooled_ses = np.zeros(D)
    pooled_p_vals = np.zeros(D)
    
    coefs = np.zeros((no_models, D))
    variances = np.zeros((no_models, D))
    
    for n, fitted_model in enumerate(fitted_models):
        coefs[n, :] = fitted_model.fitted_model.params
        variances[n, :] = np.diag(fitted_model.fitted_model.cov_params())
        
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
        
    param_list = imputed_cols.copy()
    param_list.insert(0, 'BIAS')
    
    pooled_parameters = {
        'columns': param_list,
        'pooled_coefs': pooled_coefs.tolist(),
        'pooled_ses': pooled_ses.tolist(),
        'pooled_p_vals': pooled_p_vals.tolist()
    }
    
    return pooled_parameters