# This file contains the paths to the data and results used in the project.

# Path to the current python environment
PYTHONPATH = ".venv/Scripts/python.exe"

# Dataset
DATASET_UNPROCESSED_PATH = "data/user_study/cleaned_data.csv" # Path to the unprocessed dataset
DATASET_PROCESSED_PATH = "data/user_study/data.pkl" # Path to the processed dataset

# Hyperparameter tuning
def get_path_hyperparameter_tuning_model_results(model_name, mode, eval_set):
    """Returns the path to the hyperparameter tuning results of a model."""
    return f"data/hyperparameter_tuning/{mode}/{eval_set}/{model_name}"
HYPERPARAMETER_TUNING_TUNED_HYPERPARAMETERS_PATH = ( # Path to the tuned hyperparameters
    "data/hyperparameter_tuning/tuned_hyperparameters.json"
)

# Benchmarking
def get_path_model_benchmarks(model_name, mode):
    """Returns the path to the benchmark results of a model in a given mode."""
    return f"data/training/{mode}/{model_name}/"

# Evaluation
EVALUATION_RESULTS_PATH = "data/evaluation/processed_benchmarking_results.json" # Path to the evaluation results

# Output
OUTPUT_TABLE_PATH = "data/output/tables.docx" # Path to the output tables