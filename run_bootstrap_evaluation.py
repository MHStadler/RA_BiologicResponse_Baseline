import logging
import os
import sys

from model_evaluation import perform_bootstrap_model_evaluation

if __name__ == '__main__':
    os.chdir('/mnt/jw01-aruk-home01/projects/precision_medicine_ml/EULAR_2021_submission')
    
    das_type = sys.argv[1]
    treatment = sys.argv[2]
    boot_idx = sys.argv[3]
    outcome_col = sys.argv[4]
    _type = sys.argv[5]
    
    # Occasionally the bootstrapping creates illconditioned datasets, causing the fit to fail
    # Here we simply rerun a new bootstrap until it works
    while True:
        try:
            perform_bootstrap_model_evaluation(das_type, treatment, boot_idx, outcome_col = outcome_col, _type = _type)
            break
        except:
            logging.exception('Rerunning model fit due to unexpected error')
            