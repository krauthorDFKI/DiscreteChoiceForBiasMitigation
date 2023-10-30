import tensorflow as tf
import numpy as np

from src.models.multinomial_logit import Recommender_Network as Parent

class Recommender_Network(Parent):
    
    def __init__(self, n_users, n_items, batch_size, embedding_size, l2_embs=0, n_early_stop=5, n_classes=1):
        
        super(Recommender_Network, self).__init__(n_users, n_items, batch_size, embedding_size, l2_embs, n_early_stop)
        
        self.n_classes = tf.constant(n_classes) # Anzahl der Klassen im Generalised nested logit model
        self.lmbda_logits = tf.Variable(0.5*np.ones(n_classes), dtype=tf.float32, trainable=True)
        self.alpha_logits = tf.Variable(np.log(np.random.uniform(size=(n_items, n_classes))), dtype=tf.float32, trainable=True)

    @tf.function
    def get_lmbdas(self):
        lmbdas = tf.exp(self.lmbda_logits) # lambda > 0
        return tf.maximum(lmbdas, 0.1)

    @tf.function
    def get_alphas(self):
        alphas = tf.exp(self.alpha_logits) # alpha >= 0
        return alphas / tf.reduce_sum(alphas, axis=1, keepdims=True) # sum(alpha) = 1

    @tf.function
    def gnl_loss(self, predictions, choices_pos, options):
            # Die Auswahlwahrscheinlichkeiten stammen aus: http://www.civil.northwestern.edu/trans/koppelman/PDFs/GNLTR0005082.pdf

            alpha = tf.reshape( tf.gather(self.get_alphas(), tf.reshape(options, [-1])), [tf.shape(predictions)[0], tf.shape(predictions)[1], self.n_classes] )
            lmbda = tf.tile(tf.reshape(self.get_lmbdas(), [1,1,self.n_classes]), [tf.shape(predictions)[0], tf.shape(predictions)[1],1])

            # Probability of chosing an item if the corresponding nest is selected (hier noch nicht auf summe 1 skaliert)
            p_item_givennest = tf.tile(tf.reshape(tf.exp(predictions), [tf.shape(predictions)[0], tf.shape(predictions)[1], 1]), [1,1,self.n_classes])
            p_item_givennest = tf.multiply(alpha, p_item_givennest)
            p_item_givennest = tf.math.pow(p_item_givennest, 1/lmbda)
            p_item_givennest = tf.clip_by_value(p_item_givennest, 1e-20, 1e20) # For numeric stability

            # Wahrscheinlichkeit, ein bestimmtes Nest zu wählen
            p_nest = tf.reduce_sum(p_item_givennest, axis=1)
            lmbda = tf.tile(tf.reshape(self.get_lmbdas(), [1,self.n_classes]), [tf.shape(predictions)[0],1])
            p_nest = tf.math.pow(p_nest, lmbda)
            p_nest = p_nest / tf.reduce_sum(p_nest, axis=1, keepdims=True)
            p_nest = tf.tile(tf.reshape(p_nest, [tf.shape(p_nest)[0], 1, tf.shape(p_nest)[1]]), [1,tf.shape(predictions)[1],1])

            # Wahrscheinlichkeit, ein Item zu wählen
            p_item_givennest = p_item_givennest / tf.reduce_sum(p_item_givennest, axis=1, keepdims=True)
            
            p_item = p_nest * p_item_givennest
            p_item = tf.reduce_sum(p_item, axis=2)

            loss = - tf.math.reduce_mean(tf.math.log(tf.gather(p_item, choices_pos, axis=1, batch_dims=1)))
            return loss
    
    @tf.function()
    def mf_loss(self, predictions, choices_pos, options):
        return self.gnl_loss(predictions, choices_pos, options)

    @tf.function()
    def gradient_step(self, users, options, choices_pos, optimizer):
        
        with tf.GradientTape() as tape:

            predictions = self(tf.repeat(users, tf.size(options[0])), tf.reshape(options, [-1]))
            predictions = tf.reshape(predictions, tf.shape(options))

            mf_loss = self.mf_loss(predictions, choices_pos, options)
            reg_loss = tf.math.reduce_mean(self.losses)
            loss = mf_loss + reg_loss
            
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return loss

    @tf.function()
    def test_step(self, users, options, choices_pos):
        
        predictions = self(tf.repeat(users, tf.size(options[0])), tf.reshape(options, [-1]))
        predictions = tf.reshape(predictions, tf.shape(options))
        
        loss = self.mf_loss(predictions, choices_pos, options)
        accuracy = tf.reduce_mean(tf.cast(tf.squeeze(choices_pos)==tf.math.argmax(predictions, axis=1, output_type=tf.int32), dtype=tf.float32))
        
        return loss, accuracy