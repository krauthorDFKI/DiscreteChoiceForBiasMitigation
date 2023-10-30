import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from src.config import BENCHMARKING_PLOT_LEARNING_CURVES

from src.models.binary_logit import Recommender_Network as Parent

class Recommender_Network(Parent):
    
    def __init__(self, n_users, n_items, batch_size, embedding_size, l2_embs=0, n_early_stop=5, n_negative_samples=3):
        
        self.n_negative_samples = n_negative_samples

        super(Recommender_Network, self).__init__(n_users, n_items, batch_size, embedding_size, l2_embs, n_early_stop)

        self.optimizer_class = tf.keras.optimizers.Adam

    def generate_dataset(self, data, batch_size):

        data = data.astype('int32')
            
        dataset = tf.data.Dataset.from_tensor_slices((list(data[:,0]), list(data[:,1]), list(data[:,2]))).cache().shuffle(buffer_size=1000000, reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        
        return dataset

    def sampleNegatives(self, users, choices, items_consumed_by_user, items_unconsumed_by_user):

        unique_users = np.unique(users)

        pos_users = []
        neg_users = []
        pos_items = []
        neg_items = []

        for user_idx in unique_users:
            # Sample n_negative_samples negative interactions for each positive interaction
            pos_items_one_user = items_consumed_by_user[user_idx]
            neg_items_one_user = []

            neg_items_one_user_A = np.random.choice([i for i in items_unconsumed_by_user[user_idx] if i >= 50], 
                                                    size=len(pos_items_one_user) * self.n_negative_samples, 
                                                    replace=True) 
            neg_items_one_user_B = np.random.choice([i for i in items_unconsumed_by_user[user_idx] if i < 50], 
                                                    size=len(pos_items_one_user) * self.n_negative_samples, 
                                                    replace=True) 
            
            pos_items_one_user_is_in_A = [True if i >= 50 else False for i in pos_items_one_user] * self.n_negative_samples

            neg_items_one_user = np.where(pos_items_one_user_is_in_A, neg_items_one_user_A, neg_items_one_user_B)
        
            # import random
            # for interaction_idx in range(len(pos_items_one_user)):
                # while True:
                #     if pos_items_one_user[interaction_idx] < 50:
                #         neg_item = random.choice(range(50))
                #     else:
                #         neg_item = random.choice(range(50, 100))
                #     if neg_item not in pos_items_one_user:
                #         neg_items_one_user.append(neg_item)
                #         break

            pos_users.extend([user_idx]*len(pos_items_one_user))
            neg_users.extend([user_idx]*len(neg_items_one_user))
            pos_items.extend(list(pos_items_one_user))
            neg_items.extend(list(neg_items_one_user))

        data_pos = np.stack([pos_users, pos_items, np.ones(len(pos_users))], axis=1)
        data_neg = np.stack([neg_users, neg_items, np.zeros(len(neg_users))], axis=1)

        data = np.concatenate([data_pos, data_neg])

        np.random.shuffle(data)

        return data

    def train_steps(self, data_train, data_test, learning_rate, n_epochs):
        
        tf.keras.backend.clear_session()
        
        optimizer = self.optimizer_class(learning_rate=learning_rate)
        
        train_losses = []
        test_losses = []
        test_acc = []

        min_val_loss = 0
        er_stop_count = 0
        best_weights = None
        
        from collections import defaultdict
        items_consumed_by_user_train = defaultdict(list)
        for k, v in data_train[:,[0,2]]:
            items_consumed_by_user_train[k].append(v)
        items_unconsumed_by_user_train = {}
        for key, value in items_consumed_by_user_train.items():
            items_consumed_by_user_train[key] = list(set(items_consumed_by_user_train[key])) # Remove duplicates
            items_unconsumed_by_user_train[key] = [i for i in range(100) if i not  in value]
        
        if data_test is not None:
            from collections import defaultdict
            items_consumed_by_user_test = defaultdict(list)
            for k, v in data_test[:,[0,2]]:
                items_consumed_by_user_test[k].append(v)
            items_unconsumed_by_user_test = {}
            for key, value in items_consumed_by_user_test.items():
                items_consumed_by_user_test[key] = list(set(items_consumed_by_user_test[key])) # Remove duplicates
                items_unconsumed_by_user_test[key] = [i for i in range(100) if i not in value and i not in items_consumed_by_user_train[key]]

        for epoch in range(n_epochs):            
            from time import time
            s = time()
            train_dataset = self.generate_dataset(self.sampleNegatives(data_train[:,0], data_train[:,2], items_consumed_by_user_train, items_unconsumed_by_user_train), self.batch_size)
            if data_test is not None:
                test_dataset = self.sampleNegatives(data_test[:,0], data_test[:,2], items_consumed_by_user_test, items_unconsumed_by_user_test)
            else: 
                test_dataset = None
            # print(time()-s)

            running_average_train = 0.0
            for users, items, choices in train_dataset:
                train_loss = self.gradient_step(users, items, choices, optimizer)
                if running_average_train == 0:
                    running_average_train = train_loss
                else:
                    running_average_train = 0.95 * running_average_train + (1 - 0.95) * train_loss
            train_losses.append(running_average_train)
            # print(time()-s)
            
            if test_dataset is not None:
                users_test = list(test_dataset[:,0].astype(int))
                items_test = list(test_dataset[:,1].astype(int))
                choices_test = list(test_dataset[:,2].astype(int))
                test_loss, test_accuracy= self.test_step(np.asarray(users_test), np.asarray(items_test), np.asarray(choices_test))
                test_losses.append(test_loss)
                test_acc.append(test_accuracy)

                if test_losses[-1] < min_val_loss or best_weights is None: # max_score < best_score:
                    min_val_loss = test_losses[-1]
                    er_stop_count = 0
                    
                    best_weights = self.get_weights()
                else:
                    er_stop_count += 1
                    if er_stop_count > self.n_early_stop:
                        break
            print(time()-s)

        if test_dataset is None:  
            test_losses = None
        else:
            self.set_weights(best_weights)

        if BENCHMARKING_PLOT_LEARNING_CURVES:
            _, ax1 = plt.subplots(nrows=1, ncols=2, figsize = (15, 6))
            ax1[0].plot(train_losses, label='training')
            ax1[0].plot(test_losses, label='test')
            ax1[0].set(ylabel='Loss', xlabel='Epoch', title=f'Loss over {epoch+1} epochs')
            ax1[0].legend()
            ax1[1].plot(test_acc, label='test')
            ax1[1].set(ylabel='Accuracy', xlabel='Epoch', title=f'Accuracy over {epoch+1} epochs')
            ax1[1].legend()
            plt.show()

        return np.asarray(train_losses), np.asarray(test_losses)