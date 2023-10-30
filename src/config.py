import os
import psutil
from scipy.stats import uniform
# from mango.domain.distribution import loguniform

# General
DEBUG = False
VERBOSE = False # Print additional information
BENCHMARKING_PLOT_LEARNING_CURVES = False # Plot learning curves for each model
RESET_GENERATED_DATA_ON_STARTUP = False # Delete all data from previous runs on startup

PYTHONPATH = ".venv/Scripts/python.exe"  # Path to the current python environment

assert os.path.exists(PYTHONPATH), "Invalid path to python executable"

PARALLEL_COMPUTING_N_JOBS = psutil.cpu_count() # Number of threads to use for parallel computing

# Hyperparameter tuning
HYPERPARAMETER_TUNING_BAYESIAN_MAX_NUM_ITERATIONS = 20000 # Maximum number of iterations for Bayesian hyperparameter tuning
HYPERPARAMETER_TUNING_BAYESIAN_NUM_ITERATIONS_NUM_EPOCHS = 100 # Number of benchmarks to tune the number of epochs per model

assert (
    HYPERPARAMETER_TUNING_BAYESIAN_MAX_NUM_ITERATIONS % PARALLEL_COMPUTING_N_JOBS == 0
), f"'HYPERPARAMETER_TUNING_BAYESIAN_NUM_ITERATIONS' ({HYPERPARAMETER_TUNING_BAYESIAN_MAX_NUM_ITERATIONS}) must be a multiple of the number of threads ({PARALLEL_COMPUTING_N_JOBS})"
assert (
    HYPERPARAMETER_TUNING_BAYESIAN_NUM_ITERATIONS_NUM_EPOCHS % PARALLEL_COMPUTING_N_JOBS
    == 0
), f"'HYPERPARAMETER_TUNING_BAYESIAN_NUM_ITERATIONS_NUM_EPOCHS' ({HYPERPARAMETER_TUNING_BAYESIAN_NUM_ITERATIONS_NUM_EPOCHS}) must be a multiple of the number of threads ({PARALLEL_COMPUTING_N_JOBS})"

# Benchmarking
BENCHMARKING_NUM_ITERATIONS = 500 # Number of benchmarks per model
assert (
    BENCHMARKING_NUM_ITERATIONS % PARALLEL_COMPUTING_N_JOBS == 0
), f"'TRAINING_N_RUNS' ({BENCHMARKING_NUM_ITERATIONS}) must be a multiple of the number of threads ({PARALLEL_COMPUTING_N_JOBS})"

# Hyperparameter search spaces
HYPERPARAMETER_SEARCH_SPACES = {
    "Recommender_most_popular": {}, # Most popular
    "Recommender_random": {}, # Random model
    "Recommender_knn": { # k-nearest neighbors
        "n_neighbours": range(2, 128),
    },
    "Recommender_binary_logit_negative_sampling": { # Binary logit with negative sampling
        "k": range(2, 10),
        "n_epochs": [200],
        "learning_rate": [0.01],
        "batch_size": [4 * 64],
        "l2_embs_log10": uniform(-8, 8),  # (loc, scale) -> [10^loc, 10^loc + scale]
        "n_early_stop": [20],
    },
    "Recommender_binary_logit": { # Binary logit (Binary Matrix Factorization)
        "k": range(2, 10),
        "n_epochs": [200],
        "learning_rate": [0.01],
        "batch_size": [4 * 64],
        "l2_embs_log10": uniform(-8, 8),  # (loc, scale) -> [10^loc, 10^loc + scale]
        "n_early_stop": [20],
    },
    "Recommender_multinomial_logit": { # Multinomial logit
        "k": range(2, 10),
        "n_epochs": [200],
        "learning_rate": [1],
        "batch_size": [64],
        "l2_embs_log10": uniform(-8, 8),  # (loc, scale) -> [10^loc, 10^loc + scale]
        "n_early_stop": [20],
    },
    "Recommender_exponomial": { # Exponomial logit
        "k": range(2, 10),
        "n_epochs": [200],
        "learning_rate": [1],
        "batch_size": [64],
        "l2_embs_log10": uniform(-8, 8),  # (loc, scale) -> [10^loc, 10^loc + scale]
        "n_early_stop": [20],
    },
    "Recommender_generalized_multinomial_logit": { # Generalized multinomial logit
        "k": range(2, 10),
        "n_epochs": [200],
        "learning_rate": [1],
        "batch_size": [64],
        "l2_embs_log10": uniform(-8, 8),  # (loc, scale) -> [10^loc, 10^loc + scale]
        "n_early_stop": [20],
        "n_classes": range(2, 10),
    },
    "Recommender_biser": { # BISER https://dl.acm.org/doi/pdf/10.1145/3477495.3531946 
        "k": range(
            2, 20
        ),  # Higher embedding dimension for this model than others as in the original paper
        "learning_rate": [0.1],
        "batch_size": [1],  # 1 in the original paper
        "n_epochs": [500],  # 500 in the original paper
        "l2_embs_log10": uniform(-8, 8),  # [1e-4, 14-14] in the original paper
        "wu": uniform(0.1, 0.8),  # {0.1, 0.5, 0.9} in the original paper
        "wi": uniform(0.1, 0.8),  # {0.1, 0.5, 0.9} in the original paper
        "n_early_stop": [20],
    },
    "Recommender_relmf": { # RelMF https://dl.acm.org/doi/abs/10.1145/3336191.3371783
        "k": range(2, 10),
        "learning_rate": [0.01],
        "batch_size": [256],
        "n_epochs": [500],
        "l2_embs_log10": uniform(-8, 8),  # [1e-4, 1e-14] in the BISER paper
        "clip": uniform(0.01, 0.09), # [0.01, 0.1] in the original paper
        "alpha": uniform(0.05, 0.95), # eta in the original paper, which they set to 0.5. We set it to the same as for ubpr.
        "n_early_stop": [20],
    },
    "Recommender_macr": { # MACR https://dl.acm.org/doi/pdf/10.1145/3447548.3467289
        "k": range(2, 10),
        "learning_rate": [0.01],
        "batch_size": [256],
        "n_epochs": [500],
        "l2_embs_log10": uniform(-8, 8),  # [1e-4, 1e-14] in the BISER paper
        "n_early_stop": [20],
        "macr_c": uniform(0, 10),  # {20, 22, ..., 40} in the original paper, but reduced here because our utilities vary by much less than in the original paper
        "macr_alpha_log10": uniform(
            -5, 5
        ),  #  {1e-5, 1e-4, 1e-3, 1e-2} in the original paper
        "macr_beta_log10": uniform(
            -5, 5
        ),  # {1e-5, 1e-4, 1e-3, 1e-2} in the original paper
    },
    "Recommender_pd": { # PD https://dl.acm.org/doi/pdf/10.1145/3404835.3462875
        "k": range(2, 10),
        "learning_rate": [0.01],
        "batch_size": [256],
        "n_epochs": [500],
        "l2_embs_log10": uniform(
            -8, 8
        ),  # {0, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6} in the original paper (1e-1 was not tried)
        "n_early_stop": [20],
        "alpha": [0.975],
        "gamma": uniform(0.02, 0.25),  #  {0.02, 0.04, ..., 0.25} in the original paper
    },
    "Recommender_bpr": { # Bayesian Personalized Ranking (BPR)
        "k": range(2, 10),
        "learning_rate": [0.01],
        "batch_size": [128],
        "n_epochs": [500],
        "l2_embs_log10": uniform(-8, 8),
        "n_early_stop": [20],
    },
    "Recommender_ubpr": { # Unbiased Bayesian Personalized Ranking (UBPR) https://dl.acm.org/doi/abs/10.1145/3409256.3409812
        "k": range(2, 10),
        "learning_rate": [0.001],
        "batch_size": [128],
        "n_epochs": [500],
        "l2_embs_log10": uniform(-8, 8),  # Increase maximum to 1 for baseline methods
        "n_early_stop": [20],
        "ps_pow": uniform(0.05, 0.95), # 0.05 to 1
        "clip_min": [1e-5],
    },
    "Recommender_cpr": { # CPR https://dl.acm.org/doi/pdf/10.1145/3404835.3462875
        "k": range(2, 10),
        "learning_rate": [0.003],
        "batch_size": [256],
        "n_epochs": [500],
        "l2_embs_log10": uniform(-8, 8),
        "n_early_stop": [20],
        "beta": uniform(
            1, 2
        ),  # sample rate, values in the original paper are not provided
        "max_k_interact": range(2, 4),  # {2, 3} in the original paper
        "gamma": uniform(1/3, 3),  # sampling ratio, values in the original paper are not provided
        "n_step" : [16], # Number of batches to sample per epoch
    },
}

# Names of the models in the output tables
OUTPUT_TABLES_MODEL_NAMES = {
    "Recommender_multinomial_logit": "MNL",
    "Recommender_generalized_multinomial_logit": "GEV",
    "Recommender_exponomial": "ENL",
    "Recommender_binary_logit": "BL",
    "Recommender_knn": "KNN",
    "Recommender_binary_logit_negative_sampling": "BCE",
    "Recommender_bpr": "BPR",
    "Recommender_ubpr": "UBPR",
    "Recommender_relmf": "RelMF",
    "Recommender_pd" : "PD",
    "Recommender_macr": "MACR",
    "Recommender_biser": "BISER",
    "Recommender_cpr": "CPR",
    "Recommender_most_popular": "MostPop",
    "Recommender_random": "Random",
}
assert(all([name in OUTPUT_TABLES_MODEL_NAMES for name in HYPERPARAMETER_SEARCH_SPACES])), f"Missing output names for models " + ", ".join([name for name in HYPERPARAMETER_SEARCH_SPACES if name not in OUTPUT_TABLES_MODEL_NAMES])