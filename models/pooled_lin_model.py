import numpy as np
import pickle

class PooledLinearModel():
    def __init__(self, beta0, betas):
        self.beta0 = beta0
        self.betas = betas
        
    @classmethod 
    def init_from_linear_models(cls, linear_models):
        N = len(linear_models)
        
        beta0 = linear_models[0].fitted_model.params[0]
        betas = linear_models[0].fitted_model.params[1:]
        
        for n in range(1, N):
            beta0 += linear_models[n].fitted_model.params[0]
            betas += linear_models[n].fitted_model.params[1:]
            
        beta0 /= N
        betas /= N
            
        return cls(beta0, betas)
    
    @classmethod 
    def init_from_file(cls, file_name):
        param_dict = pickle.load(open(f"./data/{file_name}.data", "rb"))
        
        beta0 = param_dict['beta0']
        betas = param_dict['betas']
        
        return cls(beta0, betas)
    
    def to_file(self, file_name):
        param_dict = {
            'beta0': self.beta0,
            'betas': self.betas
        }
        
        pickle.dump(param_dict, open(f"./data/{file_name}.data", "wb"))
    
    def predict_proba(self, X):
        x = self.lin_pred(X)
        
        return self._sigmoid(x)
    
    def lin_pred(self, X):
        return self.beta0 + np.dot(X, self.betas.T)
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))