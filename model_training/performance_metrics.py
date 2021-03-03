from models import Logit

from sklearn.metrics import brier_score_loss, roc_auc_score

def get_model_performance(pooled_lin_model, X, y):
    y_prob = pooled_lin_model.predict_proba(X)
    lin_pred = pooled_lin_model.lin_pred(X)
        
    cal_model = Logit()
    cal_model.fit(lin_pred, y)
    
    brier_score = brier_score_loss(y, y_prob)
    auc = roc_auc_score(y, y_prob)
    citl = cal_model.fitted_model.params[0]
    cal_slope = cal_model.fitted_model.params[1]
    
    return brier_score, auc, citl, cal_slope