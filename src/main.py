import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    from src.hyperparameter_tuning.tune_hyperparameters_bayesian import tune_hyperparameters
    from src.data.load_dataset import load_dataset
    from src.benchmarking.benchmark_models import benchmark_models
    from src.evaluation.process_results import process_results
    from src.output.create_output import create_output
    from src.utils import reset_generated_data
    from src.config import RESET_GENERATED_DATA_ON_STARTUP

    if RESET_GENERATED_DATA_ON_STARTUP:
        # Delete all data from previous runs
        reset_generated_data()

    # Pre-process dataset
    load_dataset()

    # Tune hyperparameters
    tune_hyperparameters()

    # Benchmark models with tuned hyperparameters
    benchmark_models()

    # Process results
    process_results()

    # Create output
    create_output()

    print("Done.")
