import pandas as pd
import random
import warnings
warnings.filterwarnings("ignore")

from autoimpute_copy.imputations import MiceImputer

bin_categories = ['BIO', 'SERO', 'SMOKE', 'SEX']

least_squares_strat = 'least squares'
binary_log_strat = 'binary logistic'
multinomial_log_strat = 'multinomial logistic'

mice_strategies = {
    'BIO': multinomial_log_strat,
    'RFPOS': binary_log_strat,
    'SMOKE': multinomial_log_strat,
    'SEX': binary_log_strat
}

imputation_regression_args = {
    'multinomial logistic': {
        'penalty': 'none',
        'solver': 'lbfgs',
        'max_iter': 8000
    },
    'binary logistic': {
        'penalty': 'none',
        'solver': 'lbfgs',
        'max_iter': 8000
    },
    'least squares': {
    }
}

def mice_impute_dataframe(dataframe, K = 4):
    shuffled_strategies = _get_shuffled_mice_strategies(dataframe.columns)
    
    mice_imputer = MiceImputer(n = 1, k = K, strategy = shuffled_strategies, imp_kwgs = imputation_regression_args, return_list = True)
    imputed_train_df = pd.DataFrame(mice_imputer.fit_transform(dataframe)[0][1])
    imputed_train_df = pd.get_dummies(imputed_train_df, drop_first = False, columns = ['BIO', 'SMOKE'])
    
    return imputed_train_df

def _get_shuffled_mice_strategies(data_columns):
    _mice_strategies = {}
    
    for column in data_columns:
        _mice_strategies[column] = mice_strategies.get(column, least_squares_strat)
        
    dict_items = list(_mice_strategies.items())
    random.shuffle(dict_items)
    
    return dict(dict_items)