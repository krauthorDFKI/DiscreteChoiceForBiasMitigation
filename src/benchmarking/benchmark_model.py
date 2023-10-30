import os
import random
import numpy as np
from datetime import datetime
import sys

def benchmark_model(model_name, mode):
    from src.hyperparameter_tuning.get_hyperparameters import get_hyperparameters
    from src.models.recommender import Recommender_most_popular, Recommender_knn, Recommender_multinomial_logit, Recommender_exponomial, Recommender_generalized_multinomial_logit, Recommender_binary_logit, Recommender_binary_logit_negative_sampling, Recommender_random, Recommender_bpr, Recommender_ubpr, Recommender_cpr, Recommender_biser, Recommender_relmf, Recommender_macr, Recommender_pd
    from src.data.load_dataset import load_dataset
    from src.utils import calc_nDCG, JSON_dump_numpy
    from src.data.split_train_val import split_train_val

    model = eval(model_name)

    data = load_dataset()
    (
        _,
        n_users,
        _,
        _,
        _,
        _,
        set_B,
        set_bias,
        _,
        preferences_from_B,
        _,
        _,
    ) = data

    if mode == "overexposure":
        eval_sets = ["B", "BIAS"]
    elif mode == "competition":
        eval_sets = ["popular", "unpopular"]
    else:
        raise ValueError("Invalid mode.")

    # Split users into training and validation sets
    all_users = [i for i in range(n_users)]
    random.shuffle(all_users)
    fold_train = all_users[: int(2/3 * len(all_users))]
    fold_validation = [i for i in range(n_users) if i not in fold_train]
    
    ### Training ###
    # For overexposure: 
    # First run: Train on data from training users (Sets A, B) and validation users (Set A)
    # Second run: Train on data from training users (Sets A, Bias) and validation users (Set A)
    # For competition: Train on data from training users (Sets A, popular alternatives (B)) and validation users (Set A)
    # First run: Train on data from training users (Sets A, popular alternatives (B)) and validation users (Set A)
    # Second run: Train on data from training users (Sets A, unpopular alternatives (Bias)) and validation users (Set A)

    train_losses = []
    rankings = []
    mean_nDCG_scores = []
    hyperparams_per_eval_set = [get_hyperparameters(model, mode, eval_set) for eval_set in eval_sets]
    
    for eval_set, hyperparams in zip(eval_sets, hyperparams_per_eval_set):

        hyperparams = get_hyperparameters(model, mode, eval_set)
        hyperparams["n_users"] = n_users
        hyperparams["n_alternatives"] = 100

        # Generate training data
        data_train, _ = split_train_val(fold_train, fold_validation, mode, eval_set)

        # Initialize model
        recommender = model(hyperparams)

        # Train model
        result_training = recommender.train(
            data_train, None, hyperparams
        )
        train_losses.append(result_training["train_losses"])

        # Review the recommendations of the validation users on Set B
        rankings.append(recommender.predict(fold_validation, set_B, rec_size=len(set_B)))

        # Calculate nDCGs
        nDCGs = []
        for u in fold_validation:
            estimated_preferences = recommender.predict(
                [u], preferences_from_B[u], rec_size=len(preferences_from_B[u])
            )
            true_preference = np.asarray([preferences_from_B[u]])

            if len(true_preference[0]) > 1:  # Not always the case
                nDCGs.append(calc_nDCG(true_preference, estimated_preferences))
        mean_nDCG_scores.append(np.mean(nDCGs))
        
        del hyperparams["n_users"]
        del hyperparams["n_alternatives"]

    ### Calculate bias coefficients ###
    mean_item_ranks = [
        np.mean(
            [[rec.tolist().index(i) for i in set_bias] for rec in ranking], axis=0
        )
        for ranking in rankings
    ]
    bias_coeffs_per_item = mean_item_ranks[0] - mean_item_ranks[1] # [0] - [1] results in a positive difference when an item is ranked higher (better) in [1] because more popular items have a lower rank

    # Store results
    result = {
        "bias_coeffs_per_item": list(bias_coeffs_per_item),
        f"mean_nDCG_{eval_sets[0]}": mean_nDCG_scores[0],
        f"mean_nDCG_{eval_sets[1]}": mean_nDCG_scores[1],
        f"hyperparameters_{eval_sets[0]}": hyperparams_per_eval_set[0],
        f"hyperparameters_{eval_sets[1]}": hyperparams_per_eval_set[1],
        f"train_loss_{eval_sets[0]}": list(train_losses[0]) if not train_losses[0] is None else None,
        f"train_loss_{eval_sets[1]}": list(train_losses[1]) if not train_losses[1] is None else None,
        "fold_train": fold_train,
    }

    # Ensure that the file name is unique
    json_path_raw = f"data/training/{mode}/{model.__name__}/{datetime.utcnow().strftime('%Y-%m-%d %H-%M-%S-%f')} {str(int(random.uniform(0, 1e-1)*1e8))}.json"
    json_path = json_path_raw
    i = 0
    while (os.path.exists(json_path)):
        i += 1
        json_path = json_path_raw + f"_{i}"

    # Save results
    JSON_dump_numpy(result, json_path)

if __name__ == "__main__":
    sys.path.append(os.getcwd())
    print(f"Params: {' '.join(sys.argv)}")
    benchmark_model(sys.argv[1], sys.argv[2])