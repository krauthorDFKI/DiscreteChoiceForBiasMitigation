import json

from src.models.recommender import *
from src.paths import HYPERPARAMETER_TUNING_TUNED_HYPERPARAMETERS_PATH

def get_hyperparameters(model, mode, eval_set):
    """Returns tuned hyperparameters for a model given the context"""
    with open(HYPERPARAMETER_TUNING_TUNED_HYPERPARAMETERS_PATH) as f:
        hyperparameters_all = json.load(f)
    return hyperparameters_all[mode][eval_set][model.__name__]

def get_all_tuned_hyperparameters():
    """Returns tuned hyperparameters for a model given the context"""
    with open(HYPERPARAMETER_TUNING_TUNED_HYPERPARAMETERS_PATH) as f:
        hyperparameters_all = json.load(f)
    return hyperparameters_all