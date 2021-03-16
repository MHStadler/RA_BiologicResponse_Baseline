import logging
import os

def create_result_directories(das_type, treatment, outcome_col = 'class_bin'):
    try:
        os.makedirs(f'./data/results/{treatment}/{outcome_col}/imputed', exist_ok = True)
        os.makedirs(f'./data/results/{treatment}/{outcome_col}/bootstraps', exist_ok = True)
    except OSError:
        logging.error('Failed to created output directories')
    else:
        logging.info('Created output directories')