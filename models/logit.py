import logging
import numpy as np
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler

# A simple Scikit Wrapper for SM.Logit
class Logit():
    def __init__(self):
        self.classes_ = [0, 1]
    
    def fit(self, X, y):
        X = sm.add_constant(X, has_constant = 'add')
        
        m = sm.Logit(y, X) 
        self.fitted_model = m.fit(maxiter = 300, cnvrg_tol = 1e-6, disp = 0)
        
    def predict_proba(self, X):
        X = sm.add_constant(X, has_constant = 'add')
        
        y_prob_preds = self.fitted_model.predict(X)
        
        return np.vstack([1 - y_prob_preds, y_prob_preds]).T
    
    def lin_pred(self, X):
        X = sm.add_constant(X, has_constant = 'add')
        
        return self.fitted_model.predict(X, linear = True)