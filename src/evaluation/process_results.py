import os
import json
import numpy as np
from tqdm import tqdm
from scipy import stats
from sklearn.linear_model import LinearRegression

from src.data.load_dataset import load_dataset
from src.models.recommender import Recommender_multinomial_logit, Recommender_most_popular, Recommender_random
from src.utils import create_path, JSON_dump_numpy
from src.hyperparameter_tuning.get_hyperparameters import get_all_tuned_hyperparameters
from src.config import BENCHMARKING_NUM_ITERATIONS
from src.paths import EVALUATION_RESULTS_PATH, get_path_model_benchmarks

def bootstrapped_two_sample_t_test(x, y, B=100000, twosided=False):
    """Performs a two-sample t-test on the difference of the means of two samples."""
    if np.mean(x) < np.mean(y):  # Mean value of x should be greater
        a = x
        x = y
        y = a

    x_bar = np.mean(x)
    y_bar = np.mean(y)
    n = len(x)
    m = len(y)
    x_var = np.var(x)
    y_var = np.var(y)

    t = (x_bar - y_bar) / np.sqrt(x_var / n + y_var / m)

    z_bar = np.mean(np.concatenate((x, y)))
    x_dash = x - x_bar + z_bar
    y_dash = y - y_bar + z_bar

    t_stars = []
    x_star = np.random.choice(x_dash, size=(B, n), replace=True)
    y_star = np.random.choice(y_dash, size=(B, m), replace=True)

    x_star_bar = np.mean(x_star, axis=1)
    y_star_bar = np.mean(y_star, axis=1)
    x_star_var = np.var(x_star, axis=1)
    y_star_var = np.var(y_star, axis=1)

    t_stars = (x_star_bar - y_star_bar) / np.sqrt(x_star_var / n + y_star_var / m)

    if twosided:
        p = np.mean(np.abs(t_stars) >= np.abs(t))  # Two-sided test
    else:
        p = np.mean(np.asarray(t_stars) >= t)  # One-sided test
    return p

def bootstrapped_one_sample_t_test(x, B=200000):
    """Performs a one-sample t-test on the difference of the means of two samples."""
    p = bootstrapped_two_sample_t_test(x, np.repeat(0, len(x)), B=B)
    return p

def calculate_bias_results(data):
    """Calculates the bias coefficients for all models."""
    result = {}
    
    for mode in data.keys():
        result[mode] = {}
        print(f"Calculating bias coefficients for {mode}...")
        for model in tqdm(data[mode].keys(), ascii=True, leave=False):
            result[mode][model] = {}

            result[mode][model]["ave_bias_coeffs_per_item"] = np.mean(data[mode][model]["bias_coeffs_per_item"], axis=0)

            # Confidence intervals of the bias coefficients per item
            result[mode][model]["bias_confidence_intervals"] = [
                stats.bootstrap(
                    (np.asarray(data[mode][model]["bias_coeffs_per_item"])[:, i],),
                    np.mean,
                    confidence_level=0.99,
                    vectorized=True,
                ).confidence_interval
                for i in range(5)
            ]

            # Check whether the bias coefficients per model and item are significantly different from 0
            p_values = []
            for i in range(5):
                p_values.append(bootstrapped_one_sample_t_test(np.asarray(data[mode][model]["bias_coeffs_per_item"])[:, i]))
            result[mode][model]["pvals_bias_coeffs_per_item"] = p_values

    return result

def calculate_performance_results(data):
    """Calculates the performance coefficients for all models."""
    result = {}
    
    for mode in data.keys():
        print(f"Calculating performance coefficients for {mode}...")

        result[mode] = {}
        
        for model in tqdm(data[mode].keys(), ascii=True, leave=False):
            result[mode][model] = {}

            if mode == "overexposure":
                eval_sets = ["B", "BIAS"]
            elif mode == "competition":
                eval_sets = ["popular", "unpopular"]

            for eval_set in eval_sets:
                result[mode][model][f"pval_nDCG_{eval_set}"] = bootstrapped_two_sample_t_test(
                    np.reshape(data[mode][model][f"mean_nDCG_{eval_set}"], -1),
                    np.reshape(data[mode][Recommender_most_popular.__name__][f"mean_nDCG_{eval_set}"], -1),
                )

    return result

def adjust_bias_results(data, bias_results):
    """Calculates the adjusted bias coefficients for all models."""
    result = {}

    (
        users,
        _,
        choices,
        options,
        set_types,
        _,
        _,
        _,
        _,
        _,
        _,
        _
    ) = load_dataset()

    print(f"Adjusting bias coefficients...")
    result = {}
    
    for model in tqdm(data["overexposure"].keys(), ascii=True, leave=False):
        result[model] = {}

        # Adjust
        # Determine difference in ranks of biased items on Set B and bias per sample
        rank_diffs = []
        rels = []
        exp_diffs = []
        for i in range(len(data["overexposure"][model]["fold_train"])):
            fold_train = data["overexposure"][model]["fold_train"][i]

            options_train_B = np.asarray(
                [
                    options[i]
                    for i in range(len(users))
                    if set_types[i] == "B" and users[i] in fold_train
                ]
            )
            choices_train_B = np.asarray(
                [
                    choices[i]
                    for i in range(len(users))
                    if set_types[i] == "B" and users[i] in fold_train
                ]
            )
            ids, counts = np.unique(choices_train_B, return_counts=True)
            pick_freqs = [counts[ids == idx][0] if idx in ids else 0 for idx in range(50)] # Handle items that were not picked
            exp_freqs = np.unique(np.reshape(options_train_B, -1), return_counts=True)[1]
            rel = pick_freqs / exp_freqs
            most_to_least_pop = [
                x for _, x in sorted(zip(rel, range(50)), reverse=True)
            ]
            ranks_B = [most_to_least_pop.index(i) for i in range(5)]
            exp_freqs_B = exp_freqs

            options_train_Bias = np.asarray(
                [
                    options[i]
                    for i in range(len(users))
                    if set_types[i] == "BIAS" and users[i] in fold_train
                ]
            )
            choices_train_Bias = np.asarray(
                [
                    choices[i]
                    for i in range(len(users))
                    if set_types[i] == "BIAS" and users[i] in fold_train
                ]
            )
            ids, counts = np.unique(choices_train_Bias, return_counts=True)
            pick_freqs = [counts[ids == idx][0] if idx in ids else 0 for idx in range(50)] # Handle items that were not picked
            exp_freqs = np.unique(np.reshape(options_train_Bias, -1), return_counts=True)[1]
            rel = pick_freqs / exp_freqs
            most_to_least_pop = [
                x for _, x in sorted(zip(rel, range(50)), reverse=True)
            ]
            ranks_Bias = [most_to_least_pop.index(i) for i in range(5)]
            exp_freqs_Bias = exp_freqs

            exp_diffs.append(np.asarray(exp_freqs_Bias) - np.asarray(exp_freqs_B))
            rank_diffs.append(np.asarray(ranks_B) - np.asarray(ranks_Bias))
            rels.append(rel)

        exp_diffs = np.asarray(exp_diffs)[:, :5]
        rank_diffs = np.asarray(rank_diffs)

        result[model]["adj_bias_coeffs_per_item"] = np.zeros(np.asarray(data["overexposure"][model]["bias_coeffs_per_item"]).shape)
        result[model]["ave_adj_bias_coeffs"] = np.zeros(bias_results["overexposure"][model]["ave_bias_coeffs_per_item"].shape)

        # Linear regression
        x = np.transpose([np.reshape(rank_diffs, -1)])
        y = np.reshape(data["overexposure"][model]["bias_coeffs_per_item"], -1)
        lg = LinearRegression(fit_intercept=True)
        lg = lg.fit(x, y)

        # Adjust coefficients
        result[model]["adj_bias_coeffs_per_item"] = np.reshape(
            y - lg.coef_[0] * x[:, 0], np.asarray(data["overexposure"][model]["bias_coeffs_per_item"]).shape
        )
        # Average adjusted bias coefficients per item
        result[model]["ave_adj_bias_coeffs"] = result[model]["adj_bias_coeffs_per_item"].mean(axis=0)

        # Adjusted confidence intervals of the bias coefficients per item
        result[model]["adj_bias_confidence_intervals"] = [
            stats.bootstrap(
                (result[model]["adj_bias_coeffs_per_item"][:, i],),
                np.mean,
                confidence_level=0.99,
                vectorized=True,
            ).confidence_interval
            for i in range(5)
        ]

        # Check whether the adjusted bias coefficients per model and item are significantly different from 0.
        p_values = []
        for i in range(5):
            p_values.append(
                bootstrapped_one_sample_t_test(result[model]["adj_bias_coeffs_per_item"][:, i])
            )
        result[model]["pvals_adj_bias_coeffs_per_item"] = p_values

    return result

def calculate_comparison_results(data, results_bias, results_bias_adjusted):
    """Calculates the p-values for the comparison of the multinomial model to all other models."""
    raise NotImplementedError("The following code chunk has not been debugged yet.")
    print("Calculating p-values for comparison to the multinomial model...")
    result = {}

    for model in tqdm(
        [
            m
            for m in results_bias["overexposure"].keys()
            if not m
            in [Recommender_multinomial_logit.__name__, 
                Recommender_most_popular.__name__, 
                Recommender_random.__name__]
        ],
        ascii=True,
    ):
        result[model] = {
            "pval_comparisonMNLMF_oexp" : [],
            "pval_adj_comparisonMNLMF_oexp" : [],
            "pval_comparisonMNLMF_unfAlts" : [],
        }

        for i in range(5):
            result[model]["pval_comparisonMNLMF_oexp"].append(
                bootstrapped_two_sample_t_test(
                    np.asarray(data["overexposure"][Recommender_multinomial_logit]["bias_coeffs_per_item"])[:, i],
                    np.asarray(data["overexposure"][model]["bias_coeffs_per_item"])[:, i],
                )
            )
            result[model]["pval_adj_comparisonMNLMF_oexp"].append(
                bootstrapped_two_sample_t_test(
                    results_bias_adjusted[Recommender_multinomial_logit]["adj_bias_coeffs_per_item"][:, i],
                    results_bias_adjusted[model]["adj_bias_coeffs_per_item"][:, i],
                )
            )
            result[model]["pval_comparisonMNLMF_unfAlts"].append(
                bootstrapped_two_sample_t_test(
                    np.asarray(data["competition"][Recommender_multinomial_logit]["bias_coeffs_per_item"])[:, i],
                    np.asarray(data["competition"][model]["bias_coeffs_per_item"])[:, i],
                )
            )

    return result

def read_data_for_models():
    """Reads the data for all models from the benchmarking results."""
    data = {}

    for mode in ["overexposure", "competition"]:
        data[mode] = {}

        for model_name in get_all_tuned_hyperparameters()["overexposure"]["B"].keys():
            result_path = get_path_model_benchmarks(model_name, mode) # f"data/training/{mode}/{model_name}/"
            sub_folders = result_path.split("/")[:-1]
            for i in range(1, len(sub_folders) + 1):
                sub_path = "/".join(sub_folders[:i])
                if not os.path.exists(sub_path):
                    os.mkdir(sub_path)

            existing_result_files = [result_path + file for _, _, files in os.walk(result_path) for file in files if os.path.splitext(file)[1] == '.json']
            existing_results = []
            for file in existing_result_files[:BENCHMARKING_NUM_ITERATIONS]: # Parallel computing rounds the number of runs so we have to cap the number of considered results here
                with open(file) as f:
                    existing_results.append(json.load(f))
                    
            data[mode][model_name] = {key: [] for key in existing_results[0].keys()}

            for result in existing_results:
                for key in data[mode][model_name].keys():
                    data[mode][model_name][key].append(result[key])

    return data

def process_results():  
    """Processes the benchmarking results."""
    print("Processing results...")

    # Results directory
    results_path = EVALUATION_RESULTS_PATH
    create_path(results_path)

    # Check for existing results
    if os.path.exists(results_path):
        print("Found existing results file. Skipping processing.")
        return

    ######## LOAD BENCHMARKS ########

    raw_results = read_data_for_models()

    ######## CALCULATE RESULTS ###########

    processed_results = {}

    processed_results["bias"] = calculate_bias_results(raw_results)

    # Test if performance significantly differs from MostPopular
    processed_results["performance"] = calculate_performance_results(raw_results)

    # Adjust bias coefficients of the items
    processed_results["bias_adjusted"] = adjust_bias_results(raw_results, processed_results["bias"])

    # Test if MNLMF is significantly better than Binary Logit on full information
    # results["comparison"] = calculate_comparison_results(data, results["bias"], results["bias_adjusted"])

    ######## REMOVE UNNECESSARY RESULTS (were used in calculate_comparison_results) ###########
    for model in processed_results["bias_adjusted"].keys():
        del processed_results["bias_adjusted"][model]["adj_bias_coeffs_per_item"]

    ######## SAVE RESULTS ###########
    JSON_dump_numpy(processed_results, results_path)
