from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# https://github.com/Jaewoong-Lee/Lee_2022_BISER/blob/main/util/preprocessor.py
def preprocess_dataset(data_train, data_test, alpha: float = 0.5) -> Tuple:
    """Load and Preprocess dataset."""

    num_users = 284
    num_items = 100

    users_train = np.unique(data_train[:,0])
    if data_test is not None:
        users_test = np.unique(data_test[:,0])

    from collections import defaultdict
    items_consumed_by_user_train = defaultdict(list)
    for k, v in data_train[:,[0,2]]:
        items_consumed_by_user_train[k].append(v)
    items_unconsumed_by_user_train = {}
    for key, value in items_consumed_by_user_train.items():
        items_consumed_by_user_train[key] = list(set(items_consumed_by_user_train[key])) # Remove duplicates
        items_unconsumed_by_user_train[key] = [i for i in range(100) if i not in value]

    if data_test is not None:
        data_train = [[user, item, 1] for user in users_train for item in items_consumed_by_user_train[user] if user not in users_test or item in range(50, 100)]
        data_train = data_train + [[user, item, 0] for user in users_train for item in items_unconsumed_by_user_train[user] if user not in users_test or item in range(50, 100)]
        data_train = np.asarray(data_train).astype(np.int32)
    else:
        data_train = [[user, item, 1] for user in users_train for item in items_consumed_by_user_train[user]]
        data_train = data_train + [[user, item, 0] for user in users_train for item in items_unconsumed_by_user_train[user]]
        data_train = np.asarray(data_train).astype(np.int32)
    
    if data_test is not None:
        items_consumed_by_user_test = defaultdict(list)
        for k, v in data_test[:,[0,2]]:
            items_consumed_by_user_test[k].append(v)
        items_unconsumed_by_user_test = {}
        for key, value in items_consumed_by_user_test.items():
            items_consumed_by_user_test[key] = list(set(items_consumed_by_user_test[key])) # Remove duplicates
            items_unconsumed_by_user_test[key] = [i for i in range(50) if i not in value and i not in items_consumed_by_user_test[key]] # Only items from list B (id <= 50)

        data_test = [[user, item, 1] for user in users_test for item in items_consumed_by_user_test[user]]
        data_test = data_test + [[user, item, 0] for user in users_test for item in items_unconsumed_by_user_test[user]]
        data_test = np.asarray(data_test).astype(np.int32)

    train = data_train
    val = data_test

    # train data freq
    item_freq = np.zeros(num_items, dtype=int)
    for tmp in train:
        if tmp[2] == 1:
            item_freq[int(tmp[1])] += 1

    # for training, only tr's ratings frequency used
    pscore_A = (item_freq[50:] / item_freq[50:].max()) ** alpha
    pscore_B = (item_freq[:50] / item_freq[:50].max()) ** alpha
    pscore = np.concatenate((pscore_B, pscore_A))

    # validation data freq
    # for testing
    if val is not None:
        for tmp in val:
            if tmp[2] == 1:
                item_freq[int(tmp[1])] += 1

    item_freq = item_freq**1.5 # pop^{(1+2)/2} gamma = 2

    # We pass the data directly instead of saving and loading it for handling parallel execution
    # save datasets
    # path_data = Path(f'./data/user_study/preprocessed/')
    # point_path = path_data / f'point_{alpha}'
    # point_path.mkdir(parents=True, exist_ok=True)

    # DO NOT SAVE AS IN THE ORIGINAL REPO BECAUSE WE RUN THE CODE IN PARALLEL
    # RETURN AND PASS INSTEAD
    # pointwise
    # np.save(file=point_path / 'train.npy', arr=train.astype(np.int32))
    # if val is not None:
    #     np.save(file=point_path / 'val.npy', arr=val.astype(np.int32))
    # else:
    #     import os
    #     if os.path.isfile(point_path / 'val.npy'):
    #         os.remove(point_path / 'val.npy')
    # np.save(file=point_path / 'pscore.npy', arr=pscore)
    # np.save(file=point_path / 'item_freq.npy', arr=item_freq)

    return train.astype(np.int32), val.astype(np.int32) if val is not None else None, pscore, item_freq
