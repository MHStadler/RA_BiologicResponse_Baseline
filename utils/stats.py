import logging
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm

from sklearn.calibration import CalibratedClassifierCV

def calc_ci(samples, confidence = 0.95):
    se = stats.sem(samples, axis = 0)
    
    return se * stats.t.ppf((1 + 0.95) / 2., samples.shape[0] - 1)