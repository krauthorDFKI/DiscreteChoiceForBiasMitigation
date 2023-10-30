import numpy as np

def split_train_val(
    batch_train, batch_val, mode, eval_set
):
    """Splits the data into training and validation sets given train and test users."""

    from src.models.recommender import prepare_data
    from src.data.load_dataset import load_dataset

    data = load_dataset()
    (
        users,
        _,
        choices,
        options,
        set_types,
        _,
        _,
        set_bias,
        _,
        _,
        prefRanksB,
        _
    ) = data

    if mode == "overexposure":

        # Train on data of the training users on sets A and either uniform or biased B
        indices_train = [
            i
            for i in range(len(users))
            if users[i] in batch_train and set_types[i] in ["A", eval_set]
        ]

        # Also train on data of the validation users on set A
        indices_validation_A = [
                i
                for i in range(len(users))
                if users[i] in batch_val and set_types[i] == "A"
            ]

        # Validate on data of the validation users on either set uniform or biased B
        indices_validation_B = [
                i
                for i in range(len(users))
                if users[i] in batch_val and set_types[i] == eval_set
            ]

    elif mode == "competition":
        # Determine which choice sets include popular and which unpopular rivals.
        options_tmp = np.asarray(
            [
                options[i]
                for i in range(len(users))
                if users[i] in batch_train
                and set_types[i] in ["B", "BIAS"]
                and np.any(options[i] >= 5)
                and np.any(options[i] < 5)
            ]
        )  # Filter out those sets that consist only of biased items.
        prefRanksB_median = np.median(
            [
                np.mean([prefRanksB[j] for j in opt if j not in set_bias])
                for opt in options_tmp
            ]
        )

        if eval_set == "popular":
            # Train on data of the training users on sets A, and choice sets with popular alternatives from B
            # Set A
            indices_train_popularAlts = [
                i
                for i in range(len(users))
                if users[i] in batch_train and set_types[i] == "A"
            ]
            # Training users' choice sets on set B, where items not from set bias are less popular than average
            indices_train_popularAlts = indices_train_popularAlts + [
                i
                for i in range(len(users))
                if users[i] in batch_train
                and set_types[i] in ["B", "BIAS"]
                and np.any(options[i] >= 5)
                and np.any(options[i] < 5)
                and np.mean([prefRanksB[j] for j in options[i] if j not in set_bias])
                <= prefRanksB_median # <=, because "best rank"=0, "worst rank"=9 -> the alternativs are more popular than the biased items
            ]  
            # Training users' choice sets on set B that do not include any items from set bias
            indices_train = indices_train_popularAlts + [
                i
                for i in range(len(users))
                if users[i] in batch_train
                and set_types[i] in ["B", "BIAS"]
                and not np.any(options[i] < 5)
            ]

            # indices_validation_A is defined below

            # Validation users' choice sets on set B, where items not from set bias are more popular than average
            indices_validation_B = [
                i
                for i in range(len(users))
                if users[i] in batch_val
                and set_types[i] in ["B", "BIAS"]
                and np.any(options[i] >= 5)
                and np.any(options[i] < 5)
                and np.mean([prefRanksB[j] for j in options[i] if j not in set_bias])
                <= prefRanksB_median # <=, because "best rank"=0, "worst rank"=9 -> the alternativs are more popular than the biased items
            ]

        elif eval_set == "unpopular":
            # Train on data of the training users on sets A, and choice sets with popular alternatives from B
            # Set A
            indices_train_unpopularAlts = [
                i
                for i in range(len(users))
                if users[i] in batch_train and set_types[i] == "A"
            ]
            # Training users' choice sets on set B, where items not from set bias are less popular than average
            indices_train_unpopularAlts = indices_train_unpopularAlts + [
                i
                for i in range(len(users))
                if users[i] in batch_train
                and set_types[i] in ["B", "BIAS"]
                and np.any(options[i] >= 5)
                and np.any(options[i] < 5)
                and np.mean([prefRanksB[j] for j in options[i] if j not in set_bias])
                > prefRanksB_median # >, because "best rank"=0, "worst rank"=9 -> the alternativs are more less than the biased items
            ]
            # Training users' choice sets on set B that do not include any items from set bias
            indices_train = indices_train_unpopularAlts + [
                i
                for i in range(len(users))
                if users[i] in batch_train
                and set_types[i] in ["B", "BIAS"]
                and not np.any(options[i] < 5)
            ]

            # indices_validation_A is defined below

            # Validation users' choice sets on set B, where items not from set bias are less popular than average
            indices_validation_B = [
                i
                for i in range(len(users))
                if users[i] in batch_val
                and set_types[i] in ["B", "BIAS"]
                and np.any(options[i] >= 5)
                and np.any(options[i] < 5)
                and np.mean([prefRanksB[j] for j in options[i] if j not in set_bias])
                > prefRanksB_median # >, because "best rank"=0, "worst rank"=9 -> the alternativs are more less than the biased items
            ]

        else:
            print(f"Unknown eval set {eval_set}")

        # Also train on data of the validation users on set A
        indices_validation_A = [
                i
                for i in range(len(users))
                if users[i] in batch_val and set_types[i] == "A"
            ]

    else:
        print(f"Unknown mode {mode}")

    users_train = users[indices_train]
    options_train = options[indices_train]
    choices_train = choices[indices_train]

    users_validation_A = users[indices_validation_A]
    options_validation_A = options[indices_validation_A]
    choices_validation_A = choices[indices_validation_A]

    users_validation_B = users[indices_validation_B]
    options_validation_B = options[indices_validation_B]
    choices_validation_B = choices[indices_validation_B]

    # Pre-process data
    data_train = prepare_data(
        np.append(users_train, users_validation_A),
        np.concatenate((options_train, options_validation_A)),
        np.append(choices_train, choices_validation_A),
    )
    data_validation = prepare_data(
        users_validation_B, options_validation_B, choices_validation_B
    )

    return data_train, data_validation