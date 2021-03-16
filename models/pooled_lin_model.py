import numpy as np
import pickle

class PooledLinearModelFactory():
    @classmethod 
    def init_from_file(cls, file_path):
        param_dict = pickle.load(open(f'{file_path}.data', "rb"))
        
        _type = param_dict['type']
        beta0 = param_dict['beta0']
        betas = param_dict['betas']
        
        if _type == 'mnlog':
            return PooledLinearMNModel(beta0, betas)
        else:
            return PooledLinearModel(beta0, betas)
        
    @classmethod 
    def init_from_linear_models(cls, linear_models, _type = 'log'):
        if _type == 'mnlog':
            return PooledLinearMNModel.init_from_linear_models(linear_models)
        else:
            return PooledLinearModel.init_from_linear_models(linear_models)

class PooledLinearModel():
    def __init__(self, beta0, betas):
        self.type = 'log'
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
    
    def predict_proba(self, X):
        x = self.lin_pred(X)
        
        return self._sigmoid(x)
    
    def lin_pred(self, X):
        return self.beta0 + np.dot(X, self.betas.T)
    
    def to_file(self, file_path):
        write_model_to_file(self, file_path)
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
# For simplicity, only supports 3 categories for now, asssuming category 0 is the base category
class PooledLinearMNModel():
    def __init__(self, beta0, betas):
        self.type = 'mnlog'
        self.beta0 = beta0
        self.betas = betas
        
    @classmethod 
    def init_from_linear_models(cls, linear_models):
        N = len(linear_models)
        
        beta0 = linear_models[0].fitted_model.params[0, :]
        betas = linear_models[0].fitted_model.params[1:, :]
        
        for n in range(1, N):
            beta0 += linear_models[n].fitted_model.params[0, :]
            betas += linear_models[n].fitted_model.params[1:, :]
            
        beta0 /= N
        betas /= N
            
        return cls(beta0, betas)  
    
    def predict_proba(self, X):
        x = self.lin_pred(X)
        
        sig_p1 = np.exp(x[:, 0])
        sig_p2 = np.exp(x[:, 1])
        
        p1 = sig_p1 / (1 + sig_p1 + sig_p2)
        p2 = sig_p2 / (1 + sig_p1 + sig_p2)
        p0 = 1 - (p1 + p2)
        
        return np.array([p0, p1, p2]).T
    
    def lin_pred(self, X):
        return self.beta0 + np.dot(X, self.betas)
    
    def to_file(self, file_path):
        write_model_to_file(self, file_path)
    
def write_model_to_file(pooled_model, file_path):
    param_dict = {
        'type':  pooled_model.type,
        'beta0': pooled_model.beta0,
        'betas': pooled_model.betas
    }
    
    pickle.dump(param_dict, open(f'{file_path}.data', "wb"))