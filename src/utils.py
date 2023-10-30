import os
import shutil
import json
import numpy as np

def calc_nDCG(true_prefs, est_prefs):
    """Calculates the normalized discounted cumulative gain (nDCG) for a given set of true and estimated preferences."""
    length = true_prefs.shape[1]
    DCG_opt = np.sum( 2**np.arange(length)[::-1]  / np.log2(2 + np.arange(length)) )
    DCG = []
    for i in range(est_prefs.shape[0]):
        DCG.append( np.sum( 2**np.asarray([length - 1 - list(true_prefs[i,:]).index(j) for j in est_prefs[i,:]])  / np.log2(2 + np.arange(length)) ) )
    nDCG = np.asarray(DCG) / DCG_opt
    return nDCG

def create_path(file_path):    
    """Creates all folders and sub-folders up to the last '/' of a geiven path"""
    sub_folders = file_path.split("/")[:-1]
    for i in range(1, len(sub_folders) + 1):
        sub_path = "/".join(sub_folders[:i])
        if not os.path.exists(sub_path):
            os.mkdir(sub_path)

def reset_generated_data():
    """Resets '/data/'"""
    print("Deleting existing generated data.")
    
    def delete_file(path):
        if os.path.exists(path):
            os.remove(path)
    def delete_dir(path):
        if os.path.exists(path):
            shutil.rmtree(path)

    delete_file("data/user_study/data.pkl")
    delete_dir("data/hyperparameter_tuning")
    delete_dir("data/training")
    delete_dir("data/evaluation")
    delete_dir("data/output")

def JSON_dump_numpy(dictionary, path, indent=None):
    """Saves a dictionary to a JSON file, converting numpy arrays to lists."""
    def convert(x):
        if hasattr(x, "tolist"):
            return x.tolist()
        raise TypeError(x)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, default=convert, indent=indent)
        pass

def JSON_dumps_numpy(dictionary, indent=None):
    """Returns a JSON string of a dictionary, converting numpy arrays to lists."""
    def convert(x):
        if hasattr(x, "tolist"):
            return x.tolist()
        raise TypeError(x)

    return json.dumps(dictionary, ensure_ascii=False, default=convert, indent=indent)