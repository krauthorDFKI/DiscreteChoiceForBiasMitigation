import numpy as np
from tqdm import tqdm
import os
import json
import pathlib
import subprocess
from mango import Tuner, scheduler
from scipy.stats import uniform
import random
import time

from src.hyperparameter_tuning.benchmark_hyperparameters import evaluate_hyperparams
from src.models.recommender import *
from src.utils import create_path
from src.hyperparameter_tuning.patch_mango import patch_mango
from src.config import (
    PYTHONPATH, 
    DEBUG, 
    HYPERPARAMETER_SEARCH_SPACES, 
    HYPERPARAMETER_TUNING_BAYESIAN_MAX_NUM_ITERATIONS, 
    HYPERPARAMETER_TUNING_BAYESIAN_NUM_ITERATIONS_NUM_EPOCHS, 
    PARALLEL_COMPUTING_N_JOBS,
    )
from src.paths import HYPERPARAMETER_TUNING_TUNED_HYPERPARAMETERS_PATH, get_path_hyperparameter_tuning_model_results

# Paper aus https://github.com/ARM-software/mango#CorePapers zitieren!!!
def tune_hyperparameters():
    """Tunes all hyperparameters"""
    print("Tuning hyperparameters...")

    patch_mango()
    
    existing_best_hyperparameters = {}

    # Check for existing results
    if os.path.exists(HYPERPARAMETER_TUNING_TUNED_HYPERPARAMETERS_PATH):
        # print("Found existing hyperparameter tuning results.")

        with open(HYPERPARAMETER_TUNING_TUNED_HYPERPARAMETERS_PATH) as f:
            existing_best_hyperparameters = json.load(f)

        existing_results_complete = True
        for mode in ["overexposure", "competition"]:        
            if mode == "overexposure":
                eval_sets = ["B", "BIAS"]
            if mode == "competition":
                eval_sets = ["popular", "unpopular"]
            for eval_set in eval_sets:
                for model_name in HYPERPARAMETER_SEARCH_SPACES.keys():
                    if existing_best_hyperparameters.get(mode, {}).get(eval_set, {}).get(model_name, None) is None:
                        existing_results_complete = False
        if existing_results_complete:
            print("Found existing hyperparameter tuning results. Skipping tuning.")
            return
        else:
            print("Found incomplete existing hyperparameter tuning results. Tuning the remaining models...")
    
    # Tuning
    best_hyperparameters = existing_best_hyperparameters.copy()

    for mode in (pbar_mode := tqdm(["overexposure", "competition"], ascii=True, position=0, leave=False)):
        pbar_mode.set_description_str(f"Mode '{mode}'")

        best_hyperparameters[mode] = best_hyperparameters.get(mode, {})

        if mode == "overexposure":
            eval_sets = ["B", "BIAS"]
        if mode == "competition":
            eval_sets = ["popular", "unpopular"]

        for eval_set in (pbar_eval_set := tqdm(eval_sets, ascii=True, position=1, leave=False)):
            pbar_eval_set.set_description_str(f"Eval set '{eval_set}'")

            best_hyperparameters[mode][eval_set] = best_hyperparameters.get(mode, {}).get(eval_set, {})

            for model_name in (pbar_model_name := tqdm(HYPERPARAMETER_SEARCH_SPACES.keys(), ascii=True, position=2, leave=False)):
                pbar_model_name.set_description_str(f"Model '{model_name}'")

                # Check for existing tuning results for this model
                if existing_best_hyperparameters.get(mode, {}).get(eval_set, {}).get(model_name, None) is not None:
                    # best_hyperparameters[mode][eval_set][model_name] = existing_best_hyperparameters[mode][eval_set][model_name]
                    pass

                # Skip models that do not need hyperparameter tuning
                elif model_name in [Recommender_most_popular.__name__, Recommender_random.__name__]:
                    best_hyperparameters[mode][eval_set][model_name] = {}

                # If the model has not been tuned yet, tune it
                else:
                    # Create folder for results
                    file_path = get_path_hyperparameter_tuning_model_results(model_name, mode, eval_set) + "/"
                    create_path(file_path)
                    
                    # Check for existing tuning results for this model
                    existing_result_files = [file_path + file for _, _, files in os.walk(file_path) for file in files if os.path.splitext(file)[1] == '.json']
                    existing_results = []
                    for file in existing_result_files:
                        with open(file) as f:
                            existing_results.append(json.load(f))

                    # Get hyperparameter grid
                    hyperparam_grid = HYPERPARAMETER_SEARCH_SPACES[model_name].copy()

                    @scheduler.parallel(n_jobs=PARALLEL_COMPUTING_N_JOBS)
                    def objective_cv(**kwargs):
                        hyperparams = kwargs
                        from src.data.load_dataset import load_dataset
                        data = load_dataset()
                        n_users = data[1]
                        all_users = [
                            i for i in range(n_users)
                        ] 
                        random.shuffle(all_users)
                        n = len(all_users) // 3
                        batches = [all_users[i:i + n] for i in range(0, len(all_users), n)][:3]

                        scores = []

                        for batch_val in batches:
                            batch_train = [user for user in all_users if user not in batch_val]
                            # Check for existing tuning results for this model
                            if DEBUG:
                                result_path = evaluate_hyperparams(
                                    model_name, 
                                    json.dumps(hyperparams), 
                                    mode, 
                                    eval_set,
                                    json.dumps(batch_train))
                            else:
                                cmd = [PYTHONPATH, 
                                        '-W ignore',
                                        pathlib.Path(__file__).parent.resolve() / 'benchmark_hyperparameters.py', 
                                        f'{model_name}',
                                        f'{json.dumps(hyperparams)}', 
                                        f'{mode}', 
                                        f'{eval_set}',
                                        f'{json.dumps(batch_train)}',
                                    ]
                                response = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                                # Check for errors
                                if len(response.stderr) > 0:
                                    raise Exception(response.stderr)
                                
                                result_path = response.stdout.split("Result path: ")[1].split("-----")[0]

                            with open(result_path) as f:
                                result = json.load(f)

                            if model_name in [Recommender_knn.__name__, Recommender_macr.__name__]: # KNN has no loss, MACR's scales in its parameters
                                scores.append(result["nDCG"])
                            else:
                                scores.append(- result["min_validation_loss"])

                        return [hyperparams] * len(scores), scores
                    
                    # Define early stopping criterion
                    def early_stop(results, optimizer, ds, X_sample, pbar): 
                        '''
                            stop if best objective does not improve for 2 iterations
                            results: dict (same keys as dict returned by tuner.minimize/maximize)
                        '''
                        # https://assets.amazon.science/00/0b/cc7247ca4edd9d480f4201af7c34/overfitting-in-bayesian-optimization-an-empirical-study-and-early-stopping-solution.pdf
                        import numpy as np
                        random_sample = ds.convert_GP_space(ds.get_domain())

                        upper_confidence_bound_sampled_and_tried = optimizer.Get_Upper_Confidence_Bound(
                            np.concatenate([X_sample, random_sample])
                        )
                        lower_confidence_bound_params_tried = optimizer.Get_Lower_Confidence_Bound(X_sample)

                        regret = upper_confidence_bound_sampled_and_tried.max() - lower_confidence_bound_params_tried.max()

                        if regret == optimizer.alpha * 2: # If kernel not yet fitted
                            regret *= 1e10 # The lower confidence bound can be above 4; prevent stopping before kernel is fitted

                        stopping_threshold = np.sqrt(optimizer.local_variance) # We assume homoscedacity here. The original paper linked above uses np.sqrt((1/3 + (1/3)/((3-1)/3)) * (np.std(CV_scores)**2)), but because we are doing 3-fold CV, our local estimate would be much noisier than their 10-fold CV estimate. This estimator suffers much less variance

                        # if results.get("early_stopping_metric", None) is None:
                        #     results["early_stopping_metric"] = []
                        # results["early_stopping_metric"].append(regret)

                        pbar.set_description(f"Regret/Stopping Threshold: {regret:.4f}/{stopping_threshold:.4f}")

                        return regret < stopping_threshold # or np.min(results["early_stopping_metric"]) < np.min(results["early_stopping_metric"][-HYPERPARAMETER_TUNING_BAYESIAN_EARLY_STOP_N_ITERATIONS:])

                    # Tune hyperparameters
                    tuner = Tuner(hyperparam_grid, 
                                  objective_cv, 
                                  {
                                      'num_iteration': int(HYPERPARAMETER_TUNING_BAYESIAN_MAX_NUM_ITERATIONS / PARALLEL_COMPUTING_N_JOBS), 
                                      'initial_random': 100, # Number of samples, not iterations -> No need to not account for the number of available threads here
                                      'early_stopping': early_stop,
                                    })

                    results = tuner.maximize()

                    # Get best hyperparameters
                    best_hyperparameters[mode][eval_set][model_name] = results['best_params']

                    # Tune the number of epochs if applicable
                    if "n_epochs" in best_hyperparameters[mode][eval_set][model_name].keys():
                        file_path = get_path_hyperparameter_tuning_model_results(model_name, mode, eval_set) + "/"

                        # Get all data generated up to this point
                        files_bayesian_tuning = [file_path + file for _, _, files in os.walk(file_path) for file in files if os.path.splitext(file)[1] == '.json']

                        # Generate HYPERPARAMETER_TUNING_BAYESIAN_NUM_ITERATIONS_NUM_EPOCHS many samples with the optimal hyperparameters
                        best_params = results['best_params']
                        best_params = {key : [val] for key, val in best_params.items()}
                        best_params["dummy"] = uniform(0,1) # Exploit mango's distributed computing framework

                        @scheduler.parallel(n_jobs=PARALLEL_COMPUTING_N_JOBS)
                        def objective(**kwargs):
                            hyperparams = kwargs
                            from src.data.load_dataset import load_dataset
                            data = load_dataset()
                            n_users = data[1]
                            all_users = [
                                i for i in range(n_users)
                            ] 
                            random.shuffle(all_users)
                            n = len(all_users) // 3
                            batches = [all_users[i:i + n] for i in range(0, len(all_users), n)][:3]

                            batch_val = batches[0]
                            batch_train = [user for user in all_users if user not in batch_val]

                            if DEBUG:
                                result_path = evaluate_hyperparams(
                                    model_name, 
                                    json.dumps(hyperparams), 
                                    mode, 
                                    eval_set,
                                    json.dumps(batch_train))
                            else:
                                cmd = [PYTHONPATH, 
                                        '-W ignore',
                                        pathlib.Path(__file__).parent.resolve() / 'benchmark_hyperparameters.py', 
                                        f'{model_name}',
                                        f'{json.dumps(hyperparams)}', 
                                        f'{mode}', 
                                        f'{eval_set}',
                                        f'{json.dumps(batch_train)}',
                                    ]
                                response = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                                # Check for errors
                                if len(response.stderr) > 0:
                                    raise Exception(response.stderr)
                                
                                result_path = response.stdout.split("Result path: ")[1].split("-----")[0]

                            with open(result_path) as f:
                                result = json.load(f)

                            if model_name in [Recommender_knn.__name__, Recommender_macr.__name__]:
                                return result["nDCG"]
                            else:
                                return result["min_validation_loss"]

                        tuner = Tuner(best_params,
                                      objective, 
                                      {
                                        'num_iteration': 0, 
                                        'initial_random': HYPERPARAMETER_TUNING_BAYESIAN_NUM_ITERATIONS_NUM_EPOCHS, 
                                        })
                        results = tuner.maximize()

                        # Get newly generated data
                        files_best_params = [file_path + file for _, _, files in os.walk(file_path) for file in files if os.path.splitext(file)[1] == '.json' 
                                            if file_path + file not in files_bayesian_tuning]

                        best_n_epochs = []
                        for file in files_best_params:
                            with open(file) as f:
                                best_n_epochs.append(json.load(f)["best_epoch_validation"])

                        # Determine best number of epochs
                        best_hyperparameters[mode][eval_set][model_name]["n_epochs"] = int(np.mean(best_n_epochs))

                # Save results
                create_path(HYPERPARAMETER_TUNING_TUNED_HYPERPARAMETERS_PATH)
                with open(HYPERPARAMETER_TUNING_TUNED_HYPERPARAMETERS_PATH, 'w', encoding='utf-8') as f:
                    json.dump(best_hyperparameters, f, ensure_ascii=False, indent=2)

                # When iterating too fast, the tqdm progress bars sometimes do not refresh properly without this
                time.sleep(0.1) 

                # Refresh progress bars
                for pbar in [pbar_mode, pbar_eval_set, pbar_model_name]:
                    pbar.refresh()
