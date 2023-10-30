import sys
import os
import random
import numpy as np
import json
from datetime import datetime
import time

def evaluate_hyperparams(
    model_name,
    hyperparams_str,
    mode,
    eval_set,
    batch_train_str = None,
):
    """Evaluates a model with given hyperparameters on a given mode and evaluation set."""

    from src.models.recommender import Recommender_random, Recommender_knn, Recommender_most_popular, Recommender_multinomial_logit, Recommender_exponomial, Recommender_generalized_multinomial_logit, Recommender_binary_logit, Recommender_biser, Recommender_relmf, Recommender_macr, Recommender_pd, Recommender_binary_logit_negative_sampling, Recommender_bpr, Recommender_ubpr, Recommender_cpr, Recommender_dice
    from src.data.load_dataset import load_dataset
    from src.utils import JSON_dump_numpy, create_path, calc_nDCG
    from src.data.split_train_val import split_train_val
    from src.paths import get_path_hyperparameter_tuning_model_results

    time_start = time.time()

    model = eval(model_name)
    hyperparams = json.loads(hyperparams_str)

    data = load_dataset()
    (
        _,
        n_users,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        preferences_from_B,
        _,
        _
    ) = data
    hyperparams["n_users"] = n_users
    hyperparams["n_alternatives"] = 100

    if batch_train_str is None:
        all_users = [
            i for i in range(n_users)
        ] 
        random.shuffle(all_users)
        batch_train = all_users[: int(2/3 * len(all_users))]
    else:
        batch_train = json.loads(batch_train_str)
    batch_val = [i for i in range(n_users) if i not in batch_train]

    # Split data into training and validation sets
    data_train, data_validation = split_train_val(batch_train, batch_val, mode, eval_set)

    # Initialize Model
    recommender = model(hyperparams)

    # Train
    result_training = recommender.train(data_train, data_validation, hyperparams)

    train_losses = result_training["train_losses"]
    validation_losses = result_training["validation_losses"]

    # Evaluate for validation users on set B to evaluate the KNN    
    nDCGs = []
    for u in batch_val:
        true_preference = np.asarray([preferences_from_B[u]])
        if len(true_preference[0]) > 1:  # Not always the case
            estimatedPreference = recommender.predict(
                [u], preferences_from_B[u], rec_size=len(preferences_from_B[u])
            )
            nDCG = calc_nDCG(true_preference, estimatedPreference)
            nDCGs.append(nDCG)
    aveNDCG = np.mean(nDCGs)

    # Choices of this user on list B
    # accs = []
    # nDCGs_choices = []
    # for u in batch_val:
    #     accs_this_user = []
    #     nDCGs_choices_this_user = []
    #     for obs in range(len(users_validation_B)):
    #         if users_validation_B[obs] == u:
    #             opt = options_validation_B[obs]
    #             estimatedPreference = recommender.predict([u], opt, rec_size=len(opt))[
    #                 0
    #             ]

    #             acc = 1 if estimatedPreference[0] == choices_validation_B[obs] else 0
    #             accs_this_user.append(acc)

    #             DCG_choice = 1 / np.log2(
    #                 2 + np.where(estimatedPreference == choices_validation_B[obs])[0][0]
    #             )
    #             nDCG_choice = DCG_choice * np.log2(2)
    #             nDCGs_choices_this_user.append(nDCG_choice)
    #     if (
    #         len(accs_this_user) > 0
    #     ):  # Some users could theoretically not have collected observations on B
    #         accs.append(np.mean(accs_this_user))
    #         nDCGs_choices.append(np.mean(nDCGs_choices_this_user))
    # aveAcc = np.mean(accs)
    # aveNDCG_choices = np.mean(nDCGs_choices)

    recommender.clear()

    # Results
    del hyperparams["n_users"]
    del hyperparams["n_alternatives"]
    result = {
        "nDCG": np.mean(aveNDCG),
        # "Accuracy": 0, np.mean(aveAcc),
        # "nDCG_choices": np.mean(aveNDCG_choices),
        "min_train_loss": np.min(train_losses),
        "min_validation_loss": np.min(validation_losses),
        "best_epoch_train": int(np.argmin(train_losses)),
        "best_epoch_validation": int(np.argmin(validation_losses)),
        "time_passed": time.time() - time_start,
        "hyperparameters": hyperparams,
        "train_losses": train_losses,
        "validation_losses": validation_losses,
    }

    # Store results
    json_path_raw = get_path_hyperparameter_tuning_model_results(model_name, mode, eval_set) + f"/{datetime.utcnow().strftime('%Y-%m-%d %H-%M-%S-%f')} {str(int(random.uniform(0, 1e-1)*1e8))}.json"
    json_path = json_path_raw
    i = 0
    while (os.path.exists(json_path)):
        i += 1
        json_path = json_path_raw + f"_{i}"
    create_path(json_path)
    JSON_dump_numpy(result, json_path)

    print(f"-----Result path: {json_path}-----")
    return json_path

if __name__ == "__main__":
    sys.path.append(os.getcwd())
    print(f"Params: {' '.join(sys.argv)}")
    if len(sys.argv) == 5:
        evaluate_hyperparams(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) == 6:
        evaluate_hyperparams(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])