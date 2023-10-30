import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, regularizers
tfkl = tf.keras.layers
import matplotlib.pyplot as plt
from src.config import BENCHMARKING_PLOT_LEARNING_CURVES

class Recommender_Network(Model):
    
    def __init__(self, n_users, n_items, batch_size, embedding_size, l2_embs=0, n_early_stop=5):
        """Initialize the Recommender Network"""

        super(Recommender_Network, self).__init__()
        
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.l2_embs   = l2_embs
        self.n_early_stop = n_early_stop

        self.optimizer_class = tf.keras.optimizers.SGD
        
        self.user_emb = tfkl.Embedding(input_dim=n_users, output_dim=embedding_size, trainable=True, embeddings_regularizer=regularizers.l2(self.l2_embs))
        self.item_emb = tfkl.Embedding(input_dim=n_items, output_dim=embedding_size, trainable=True, embeddings_regularizer=regularizers.l2(self.l2_embs))
        self.bias = tfkl.Embedding(input_dim=n_items, output_dim=1, trainable=True)

        self.out = tfkl.Dense(units=1)
        
    @tf.function()
    def call(self, users, items):
        """Feed input through the network layer by layer"""        
        
        user_embedding = self.user_emb(users)
        item_embedding = self.item_emb(items)

        bias = self.bias(items)

        out = tf.reduce_sum(tf.multiply(user_embedding, item_embedding), axis=1)
        
        out = out + tf.reshape(bias, out.shape)

        return out
    
    @tf.function()
    def mf_loss(self, predictions, choices_pos):
        return - tf.math.reduce_mean(tf.math.log(tf.gather(tf.nn.softmax(predictions), choices_pos, batch_dims=1)))

    def generate_dataset(self, users, options, choices, batch_size):
        """Function to time the duration of each epoch"""
            
        user = list(np.asarray(users).astype('int32'))
        options_list = list(np.array(options.tolist()).astype('int32'))
        choice_pos = [[np.where(options[i]==choices[i])[0][0].astype('int32')] for i in range(len(users))]

        dataset = tf.data.Dataset.from_tensor_slices((user, options_list, choice_pos)).cache().shuffle(buffer_size=1000000, reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        
        return dataset

    @tf.function()
    def gradient_step(self, users, options, choices_pos, optimizer):
        """Perform a training step for the given model"""
        
        with tf.GradientTape() as tape:

            predictions = self(tf.repeat(users, tf.size(options[0])), tf.reshape(options, [-1]))
            predictions = tf.reshape(predictions, tf.shape(options))

            mf_loss = self.mf_loss(predictions, choices_pos)
            reg_loss = tf.math.reduce_mean(self.losses)
            loss = mf_loss + reg_loss
            
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return loss

    @tf.function()
    def test_step(self, users, options, choices_pos):
        """Tests the models loss over the given data set"""
        
        predictions = self(tf.repeat(users, tf.size(options[0])), tf.reshape(options, [-1]))
        predictions = tf.reshape(predictions, tf.shape(options))
        
        loss = self.mf_loss(predictions, choices_pos)
        accuracy = tf.reduce_mean(tf.cast(tf.squeeze(choices_pos)==tf.math.argmax(predictions, axis=1, output_type=tf.int32), dtype=tf.float32))
        
        return loss, accuracy

    def train_steps(self, train_dataset, test_dataset, learning_rate, n_epochs):
        """Trains the given model"""  
        
        tf.keras.backend.clear_session()
        
        optimizer = self.optimizer_class(learning_rate=learning_rate) # Adam(learning_rate=learning_rate)
        
        train_losses = []
        test_losses = []
        test_acc = []

        min_val_loss = 0
        er_stop_count = 0
        best_weights = None
        
        for epoch in range(n_epochs):
            
            # Training 
            # from time import time
            # s = time()
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
                users_test = []
                items_test = []
                choices_test = []
                for users, items, choices in test_dataset:
                    users_test.extend(list(users.numpy()))
                    items_test.extend(list(items.numpy()))
                    choices_test.extend(list(choices.numpy()))
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
            # print(time()-s)

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