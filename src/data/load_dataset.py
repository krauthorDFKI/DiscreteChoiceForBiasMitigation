import pandas as pd
import numpy as np
import ast
import scipy.stats as stats
import pickle
import os

from src.config import VERBOSE
from src.paths import DATASET_UNPROCESSED_PATH, DATASET_PROCESSED_PATH

def splitLettersAndNumbers(s):
    """Splits a string into a head and a tail, where the tail is the last number in the string."""
    head = s.rstrip('0123456789')
    tail = s[len(head):]
    return head, tail

def load_dataset():
    """Loads the dataset from the csv file and processes it into a tuple of numpy arrays."""
    # Process the dataset if it has not been processed yet
    if not os.path.exists(DATASET_PROCESSED_PATH):
        if VERBOSE:
            print("No pre-processed data file found.")
            print("Processing data.")
        
        # Load the dataset from the csv file
        df = pd.read_csv(DATASET_UNPROCESSED_PATH, sep=",")

        # Exclude test runs
        df = df[df["player.field"] != "test"]
        df = df[df["player.field"] != "789"]
        df = df[df["player.field"] != "Test"]

        df = df[df["player.preferences_from_B"].notna()]

        # Exclude users who are too fast (selections)
        part_codes = np.unique(df["participant.code"])
        df_pageTimes = pd.read_csv("data/user_study/page_times.csv", sep=",")

        if VERBOSE:
            print(f"{len(df)} participants completed the questionaire.")

        # Exclude users who are too fast (selections)
        times_started = []
        times_finished = []
        for pc in part_codes:
            times_started.append(
                df_pageTimes[
                    (df_pageTimes["participant_code"] == pc)
                    & (df_pageTimes["page_name"] == "Preface")
                ]["epoch_time_completed"]
            )
            times_finished.append(
                df_pageTimes[
                    (df_pageTimes["participant_code"] == pc)
                    & (df_pageTimes["page_name"] == "CoursesSelect39")
                ]["epoch_time_completed"]
            )
        times_started = np.reshape(np.asarray(times_started), -1)
        times_finished = np.reshape(np.asarray(times_finished), -1)
        durs = times_finished - times_started
        sus_part_codes = part_codes[durs < 160]

        if VERBOSE:
            print(
                f"{len(sus_part_codes)} participants took less than four seconds for choosing."
            )

        for suspc in sus_part_codes:
            df = df[df["participant.code"] != suspc]

        if VERBOSE:
            print(f"{len(df)} participants took at least four seconds for choosing.")

        # Exclude users who are too fast (preferences)
        part_codes = np.unique(df["participant.code"])
        df_pageTimes = pd.read_csv("data/user_study/page_times.csv", sep=",")
        times_started = []
        times_finished = []

        for pc in part_codes:
            times_started.append(
                df_pageTimes[
                    (df_pageTimes["participant_code"] == pc)
                    & (df_pageTimes["page_name"] == "Preferences")
                ]["epoch_time_completed"].values[0]
            )
            times_finished.append(
                df_pageTimes[
                    (df_pageTimes["participant_code"] == pc)
                    & (df_pageTimes["page_name"] == "Preferences_from_B")
                ]["epoch_time_completed"].values[0]
            )

        times_started = np.reshape(np.asarray(times_started), -1)
        times_finished = np.reshape(np.asarray(times_finished), -1)
        durs = times_finished - times_started

        # Should take a maximum of 10 seconds for the second ranking task
        sus_part_codes = part_codes[durs < 10]

        if VERBOSE:
            print(f"{len(sus_part_codes)} participants took less than ten seconds for ranking.")

        for suspc in sus_part_codes:
            df = df[df["participant.code"] != suspc]

        if VERBOSE:
            print(f"{len(df)} participants took at least ten seconds for ranking.")

        # Do some preprocessing
        preferences = df["player.preferences"].values.tolist()
        preferences_from_B = df["player.preferences_from_B"].values.tolist()

        trial_cols = [col for col in df.columns if "coursestest" in col]
        attention_task_responses = df["player.TrickQuestion"].values.tolist()
        prep = df[trial_cols]

        new = pd.DataFrame().reindex_like(prep)
        new_sets = pd.DataFrame().reindex_like(prep)

        for col in prep.columns:
            new[col] = prep[col].str.split("[", expand=True)[0]
            new_sets[col] = "[" + prep[col].str.split("[", expand=True)[1]

        choices = new.T
        options = new_sets.T

        set_types = []
        choices = np.array(choices.T).flatten()

        choices_new = []
        for _, choice in enumerate(choices):
            word, number = splitLettersAndNumbers(choice)
            set_types.append(word)
            number = ast.literal_eval(number)
            choices_new.append(number)

        preferences_new = []
        for preference in preferences:
            preferences_new.append(ast.literal_eval(preference))

        preferences_from_B_new = []
        for preference in preferences_from_B:
            preferences_from_B_new.append(ast.literal_eval(preference))

        options = np.array(options.T)
        set_types = np.array(set_types)
        choices = np.array(choices_new)
        preferences = np.array(preferences_new, dtype=object)
        preferences = np.asarray(
            [[i for i in l if i < 50] for l in preferences], dtype=object
        )  # if the preferences only came from Set B
        preferences = np.asarray(
            [pd.unique(p) for p in preferences], dtype=object
        )  # pd is important here because, unlike numpy, it does not sort

        preferences_from_B = np.array(preferences_from_B_new, dtype=object)
        preferences_from_B = np.asarray(
            [[i for i in l if i < 50] for l in preferences_from_B], dtype=object
        )
        preferences_from_B = np.asarray(
            [pd.unique(p) for p in preferences_from_B], dtype=object
        )

        new_options = []
        for user_options in options:
            new_user_options = []

            for option in user_options:
                new_user_options.append(ast.literal_eval(option))

            new_options.append(new_user_options)

        options = np.array(new_options)
        options = options.reshape(-1, options.shape[-1])
        options = options.astype(int)

        preferences_from_B = preferences_from_B.astype(int)
        attention_task_responses = np.asarray(attention_task_responses)
        attention_task_responses_perChoice = np.repeat(
            attention_task_responses, len(choices) / len(preferences)
        )

        # Exclude users who failed the attention task
        if VERBOSE:
            print(
                f"{sum(attention_task_responses==False)} participants failed the attention task."
            )
            print(
                f"{sum(attention_task_responses==True)} participants passed the attention task."
            )

        choices = choices[attention_task_responses_perChoice]
        options = options[attention_task_responses_perChoice]
        set_types = set_types[attention_task_responses_perChoice]
        preferences = preferences[attention_task_responses]
        preferences_from_B = preferences_from_B[attention_task_responses]
        attention_task_responses_perChoice = attention_task_responses_perChoice[
            attention_task_responses_perChoice
        ]

        n_users = len(preferences)
        users = np.repeat(range(len(preferences)), len(choices) / len(preferences))

        # Test for position bias:
        pos = []
        for i in range(len(users)):
            c = choices[i]
            o = list(options[i])
            pos.append(o.index(c))

        lplace = stats.chisquare(
            f_obs=np.unique(pos, return_counts=True)[1],
            f_exp=[len(pos) / 4] * 4,
            ddof=0,
            axis=0,
        )

        if lplace.pvalue > 0.1:
            if VERBOSE:
                print(f"Keinen Position-Bias erkannt.")
        else:
            if VERBOSE:
                print(f"Position-Bias erkannt.")

        # Exclude users who choose more non-uniformly than likely
        choice = [
            [np.where(options[i] == choices[i])[0][0].astype("int32")]
            for i in range(len(users))
        ]

        choices_per_user = np.reshape(choice, (int(len(choice) / 40), 40))
        devs = []
        chiSquare_pVals = []
        dist_peruser = []

        for u in range(n_users):
            dist_thisuser = np.unique(choices_per_user[u], return_counts=True)[1]
            devs.append(np.std(dist_thisuser))

            chiSquare_pVals.append(
                stats.chisquare(
                    f_obs=dist_thisuser, f_exp=[10, 10, 10, 10], ddof=0, axis=0
                )[1]
            )

            dist_peruser.append(dist_thisuser)

        users_too_uniform = np.where(np.asarray(chiSquare_pVals) < 0.05)[0]
        if VERBOSE:
            print(f"{len(users_too_uniform)} participants likely did not choose uniformly.")

        users_not_too_uniform = [
            True if users[i] not in users_too_uniform else False for i in range(len(users))
        ]

        if VERBOSE:
            print(f"{len(users_too_uniform)} participants likely chose uniformly.")

        # Filter by attention_task_responses
        choices = choices[users_not_too_uniform]
        options = options[users_not_too_uniform]
        set_types = set_types[users_not_too_uniform]

        preferences = preferences[[i for i in range(n_users) if i not in users_too_uniform]]

        preferences_from_B = preferences_from_B[
            [i for i in range(n_users) if i not in users_too_uniform]
        ]

        n_users = len(preferences)
        users = np.repeat(range(len(preferences)), len(choices) / len(preferences))

        pref_ranks_B = []
        for p in preferences_from_B:
            pref_ranks_B.append([list(p).index(i) if i in p else 0 for i in range(0, 50)])
        pref_ranks_B = np.asarray(pref_ranks_B)
        pref_ranks_B = np.nanmean(np.where(pref_ranks_B != 0, pref_ranks_B, np.nan), axis=0)

        n_items = 100
        n_biased_items = 5
        set_A = [i for i in range(int(n_items / 2), n_items)]
        set_B = [i for i in range(n_items) if i not in set_A]
        set_bias = [i for i in range(n_biased_items)]  # Items that are unfairly favoured
        T = None

        data = (
            users,
            n_users,
            choices,
            options,
            set_types,
            set_A,
            set_B,
            set_bias,
            preferences,
            preferences_from_B,
            pref_ranks_B,
            T,
        )

        with open(DATASET_PROCESSED_PATH, 'wb') as file:
            pickle.dump(data, file)
        if VERBOSE:
            print("Processed and stored data.")

    # Load the tuple from the pickle file
    if VERBOSE:
        print("Loading pre-processed data file.")

    with open(DATASET_PROCESSED_PATH, 'rb') as file:
        data = pickle.load(file)

    if VERBOSE:
        print("Loaded data.")
    
    return data
