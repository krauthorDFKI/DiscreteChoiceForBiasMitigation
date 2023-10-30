import os
from tqdm import tqdm
import pathlib
import subprocess
from mango import Tuner, scheduler
from scipy.stats import uniform
import time

from src.utils import create_path
from src.benchmarking.benchmark_model import benchmark_model
from src.hyperparameter_tuning.get_hyperparameters import get_all_tuned_hyperparameters
from src.hyperparameter_tuning.patch_mango import patch_mango
from src.config import BENCHMARKING_NUM_ITERATIONS, DEBUG, PARALLEL_COMPUTING_N_JOBS
from src.paths import PYTHONPATH, get_path_model_benchmarks

def benchmark_models():
    """Trains all models."""
    print("Benchmarking...")

    patch_mango()

    # Check for existing results
    num_existing_results = {}
    existing_results_complete = True
    some_results_available = False
    for mode in ["overexposure", "competition"]:        
        num_existing_results[mode] = {}

        for model_name in get_all_tuned_hyperparameters()["overexposure"]["B"].keys():

            result_path = get_path_model_benchmarks(model_name, mode) # f"data/training/{mode}/{model_name}/"
            if os.path.exists(result_path):
                existing_result_files = [result_path + file for _, _, files in os.walk(result_path) for file in files if os.path.splitext(file)[1] == '.json']
                num_existing_results[mode][model_name] = len(existing_result_files)
            else:
                num_existing_results[mode][model_name] = 0

            if num_existing_results[mode][model_name] >= BENCHMARKING_NUM_ITERATIONS:
                some_results_available = True
                # print(f"Found existing benchmarks for model '{model_name}' in mode '{mode}'.")
            elif BENCHMARKING_NUM_ITERATIONS > num_existing_results[mode][model_name] > 0:
                some_results_available = True
                # print(f"Found {num_existing_results[mode][model_name]}/{BENCHMARKING_NUM_ITERATIONS} existing benchmarks for model '{model_name}' in mode '{mode}'.")
                existing_results_complete = False
            else:
                existing_results_complete = False
    
    if existing_results_complete:
        print("Found enough existing benchmarks. Skipping benchmarking.")
        return
    elif some_results_available:
        print("Found incomplete existing benchmarks. Benchmarking the remaining models...")

    # Benchmark models
    for mode in (pbar_mode := tqdm(["overexposure", "competition"], ascii=True, position=0, leave=False)):
        pbar_mode.set_description_str(f"Mode {mode}")

        for model_name in (pbar_model_name := tqdm(get_all_tuned_hyperparameters()["overexposure"]["B"].keys(), ascii=True, position=1, leave=False)):
            pbar_model_name.set_description_str(f"Model {model_name}")

            # Results directory
            result_path = get_path_model_benchmarks(model_name, mode) # f"data/training/{mode}/{model_name}/"
            create_path(result_path)

            # Check for available results for this model
            existing_result_files = [result_path + file for _, _, files in os.walk(result_path) for file in files if os.path.splitext(file)[1] == '.json']

            # Train model if not enough results are available
            if num_existing_results[mode][model_name] < BENCHMARKING_NUM_ITERATIONS:

                @scheduler.parallel(n_jobs=PARALLEL_COMPUTING_N_JOBS)
                def objective(**kwargs):
                    if DEBUG:
                        benchmark_model(model_name, mode)
                    else:
                        cmd = [PYTHONPATH, 
                                '-W ignore',
                                pathlib.Path(__file__).parent.resolve() / 'benchmark_model.py', 
                                f'{model_name}',
                                f'{mode}',
                        ]
                        response = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                        # Check for errors
                        if len(response.stderr) > 0:
                            raise Exception(response.stderr)
                        
                    return kwargs["dummy"]
                
                tuner = Tuner({"dummy": uniform(0, 1)}, 
                                objective, 
                                {
                                    'num_iteration': 0,
                                    'initial_random': BENCHMARKING_NUM_ITERATIONS - num_existing_results[mode][model_name],
                                })
                tuner.maximize() # eigentlich sollte run_initial genügen. Wäre auch deutlich schneller

            # When iterating too fast, the tqdm progress bars sometimes do not refresh properly without this
            time.sleep(0.1) 

            # Refresh progress bars
            for pbar in [pbar_mode, pbar_model_name]:
                pbar.refresh()