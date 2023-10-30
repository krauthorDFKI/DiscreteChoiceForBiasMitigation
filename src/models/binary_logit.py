import tensorflow as tf
import numpy as np

from src.models.multinomial_logit import Recommender_Network as Parent

class Recommender_Network(Parent):
    
    def __init__(self, n_users, n_items, batch_size, embedding_size, l2_embs=0, n_early_stop=5):

        super(Recommender_Network, self).__init__(n_users, n_items, batch_size, embedding_size, l2_embs, n_early_stop)

        self.optimizer_class = tf.keras.optimizers.Adam
    
    def generate_dataset(self, data, batch_size):

        data = data.astype('int32')
        
        user = []
        item = []
        rating = []

        for us in range(len(data)):
            user.append(data[us,0].astype('int32'))
            item.append(data[us,1].astype('int32'))
            rating.append(data[us,2].astype('int32'))
            
        dataset = tf.data.Dataset.from_tensor_slices((user, item, rating)).cache().shuffle(buffer_size=1000000, reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        
        return dataset

    @tf.function()
    def mf_loss(self, ratings, predictions):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)(ratings, predictions)
        
    @tf.function()
    def gradient_step(self, users, items, ratings, optimizer):
        
        with tf.GradientTape() as tape:
            predictions = self(users, items)
            mf_loss = self.mf_loss(ratings, predictions)
            reg_loss = tf.math.reduce_mean(self.losses)
            loss = mf_loss + reg_loss
            
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return loss

    @tf.function()
    def test_step(self, users, items, ratings):
        
        predictions = self(users, items)
        loss = self.mf_loss(ratings, predictions)
        
        probs = 1 / (1 + tf.exp(-predictions))
        accuracy = tf.reduce_mean(tf.cast(tf.cast(probs  > 0.5, tf.bool) == tf.cast(ratings, tf.bool), tf.float32))
        
        return loss, accuracy