import logging
import numpy as np
import statsmodels.api as sm

# A simple Scikit Wrapper for SM.OLS
class LinearModel():
    def fit(self, X, y):
        X = sm.add_constant(X, has_constant = 'add')
        
        m = sm.OLS(y, X) 
        self.fitted_model = m.fit(maxiter = 300, cnvrg_tol = 1e-6, disp = 0)
        
    def predict(self, X):
        X = sm.add_constant(X, has_constant = 'add')
        
        return self.fitted_model.predict(X)