import numpy as np
import pickle

class PooledScaler():
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance
        
    @classmethod 
    def init_from_scalers(cls, scalers):
        N = len(scalers)
        
        mean = scalers[0].mean_
        variance = scalers[0].var_
        
        for n in range(1, N):
            mean += scalers[n].mean_
            variance += scalers[n].scale_
            
        mean /= N
        variance /= N
            
        return cls(mean, variance)
    
    @classmethod 
    def init_from_file(cls, file_name):
        param_dict = pickle.load(open(f"./data/{file_name}.data", "rb"))
        
        mean = param_dict['mean']
        variance = param_dict['variance']
        
        return cls(mean, variance)
    
    def transform(self, X):
        return (X - self.mean) / np.sqrt(self.variance)
    
    def to_file(self, file_name):
        param_dict = {
            'mean': self.mean,
            'variance': self.variance
        }
        
        pickle.dump(param_dict, open(f"./data/{file_name}.data", "wb"))
        