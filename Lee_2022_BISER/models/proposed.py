
import tensorflow as tf
import numpy as np
from scipy import sparse

from Lee_2022_BISER.models.recommenders import AbstractRecommender
from Lee_2022_BISER.evaluate.evaluator import aoa_evaluator, unbiased_evaluator

from src.config import VERBOSE

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tocsr(data: np.array, num_user: int, num_item: int) -> sparse.csr_matrix:
    """Convert data to csr_matrix."""
    matrix = sparse.lil_matrix((num_user, num_item))
    for (u, i, r) in data[:, :3]:
        matrix[u, i] = r
    return sparse.csr_matrix(matrix)

def csr_to_user_dict_i(train_matrix):
    """convert a scipy.sparse.csr_matrix to a dict,
    where the key is row number, and value is the
    non-empty index in each row.
    """
    train_matrix = train_matrix.transpose()
    train_dict = {}
    for idx, value in enumerate(train_matrix):
        train_dict[idx] = value.indices.copy().tolist()
    return train_dict

def csr_to_user_dict_u(train_matrix):
    """convert a scipy.sparse.csr_matrix to a dict,
    where the key is row number, and value is the
    non-empty index in each row.
    """
    train_dict = {}
    for idx, value in enumerate(train_matrix):
        train_dict[idx] = value.indices.copy().tolist()
    return train_dict


def l2_loss(*params):
    return tf.add_n([tf.nn.l2_loss(w) for w in params])


class Proposed(AbstractRecommender):
    def __init__(self, sess, data, train, val, test, num_users: np.array, num_items: np.array, \
                 hidden_dim: int, eta: float, reg: float, max_iters: int, batch_size: int, \
                 random_state: int, wu=0.1, wi=0.1) -> None:

        """Initialize Class."""
        self.data = data
        self.num_users = num_users
        self.num_items = num_items

        self.hidden_dim_u = hidden_dim
        self.hidden_dim_i = hidden_dim
        self.eta_u = eta
        self.eta_i = eta
        self.reg_u = reg
        self.reg_i = reg
        self.batch_size_u = batch_size
        self.batch_size_i = batch_size

        self.best_weights_enc_u = None
        self.best_weights_dec_u = None
        self.best_bias_enc_u = None
        self.best_bias_dec_u = None

        self.best_weights_enc_i = None
        self.best_weights_dec_i = None
        self.best_bias_enc_i = None
        self.best_bias_dec_i = None

        self.num_epochs = max_iters
        self.train = train
        self.val = val     

        self.test = test
        self.wu = wu
        self.wi = wi

        self.train_ui_matrix = tocsr(train, num_users, num_items).toarray()
        self.train_iu_matrix = np.copy( self.train_ui_matrix.T )

        # Validation set
        if not self.val is None:
            self.val_ui_matrix = tocsr(self.val, self.num_users, self.num_items).toarray() + self.train_ui_matrix # + self.train_ui_matrix to provide the autoencoders with info on what happened on set A
            self.val_iu_matrix = np.copy( self.val_ui_matrix.T )

        self.sess = sess
        self.random_state = random_state

        self.model_name = 'proposed'

        # Build the graphs
        self.create_placeholders()
        self.build_graph()

        self.create_losses()
        self.add_optimizer()


    def create_placeholders(self):
        with tf.name_scope("input_data"):
            self.input_R_i = tf.compat.v1.placeholder(tf.float32, [None, self.num_users])
            self.pscore_i = tf.compat.v1.placeholder(tf.float32, [None, self.num_users]) # not used
            self.iAE_input_i = tf.compat.v1.placeholder(tf.float32, [None, self.num_users])
            self.w_i = tf.compat.v1.placeholder(tf.float32)

            self.input_R_u = tf.compat.v1.placeholder(tf.float32, [None, self.num_items])
            self.pscore_u = tf.compat.v1.placeholder(tf.float32, [None, self.num_items]) # not used
            self.iAE_input_u = tf.compat.v1.placeholder(tf.float32, [None, self.num_items])
            self.w_u = tf.compat.v1.placeholder(tf.float32)

            self.ignoreSetB_mask = tf.compat.v1.placeholder(tf.bool, [None, self.num_items])
            self.ignoreTestUsers_mask = tf.compat.v1.placeholder(tf.bool, [None, self.num_users])

            self.val_target_R_i = tf.compat.v1.placeholder(tf.float32, [None, self.num_users])
            self.val_target_R_u = tf.compat.v1.placeholder(tf.float32, [None, self.num_items])


    def build_graph(self):
        with tf.name_scope("embedding_layer_i"):  # The embedding initialization is unknown now
            initializer = tf.initializers.GlorotUniform(seed=self.random_state) # Previously in TF 1.X: tf.contrib.layers.xavier_initializer(seed=self.random_state)
            self.weights_i = {'encoder': tf.Variable(initializer([self.num_users, self.hidden_dim_i])),
                            'decoder': tf.Variable(initializer([self.hidden_dim_i, self.num_users]))}
            self.biases_i = {'encoder': tf.Variable(initializer([self.hidden_dim_i])),
                           'decoder': tf.Variable(initializer([self.num_users]))}

        with tf.name_scope("embedding_layer_u"):  # The embedding initialization is unknown now
            initializer = tf.initializers.GlorotUniform(seed=self.random_state) # Previously in TF 1.X: tf.contrib.layers.xavier_initializer(seed=self.random_state)
            self.weights_u = {'encoder': tf.Variable(initializer([self.num_items, self.hidden_dim_u])),
                            'decoder': tf.Variable(initializer([self.hidden_dim_u, self.num_items]))}
            self.biases_u = {'encoder': tf.Variable(initializer([self.hidden_dim_u])),
                           'decoder': tf.Variable(initializer([self.num_items]))}


        with tf.name_scope("prediction_i"):
            corrupted_input_i = self.input_R_i # no mask
            self.encoder_op_i = tf.sigmoid(tf.matmul(corrupted_input_i, self.weights_i['encoder']) +
                                                  self.biases_i['encoder'])
            
            self.decoder_op_i = tf.matmul(self.encoder_op_i, self.weights_i['decoder']) + self.biases_i['decoder']
            self.output_i = tf.sigmoid(self.decoder_op_i)

        with tf.name_scope("prediction_u"):
            corrupted_input_u = self.input_R_u # no mask
            self.encoder_op_u = tf.sigmoid(tf.matmul(corrupted_input_u, self.weights_u['encoder']) +
                                                  self.biases_u['encoder'])
            
            self.decoder_op_u = tf.matmul(self.encoder_op_u, self.weights_u['decoder']) + self.biases_u['decoder']
            self.output_u = tf.sigmoid(self.decoder_op_u)


    def create_losses(self):
        with tf.name_scope("loss"):
            eps = 0.00001

            # iAE update loss
            # ppscore_i = tf.clip_by_value(self.output_i, clip_value_min=0.1, clip_value_max = 1 )
            # self.loss_self_unbiased_i = tf.reduce_sum( self.input_R_i/ppscore_i * tf.square(1 - self.output_i) + (1 - self.input_R_i/ppscore_i) * tf.square(self.output_i) )
            # self.loss_i_u_pos_rel_i = tf.reduce_sum( self.input_R_i * tf.square(self.iAE_input_i - self.output_i)  )

            # Mask the train loss for validation users and items from Set B
            self.ignoreTestUsers_mask = tf.cast(self.ignoreTestUsers_mask, tf.int32)

            ppscore_i = tf.clip_by_value(self.output_i, clip_value_min=0.1, clip_value_max = 1 )
            loss_self_unbiased_i_pointwise = self.input_R_i/ppscore_i * tf.square(1 - self.output_i) + (1 - self.input_R_i/ppscore_i) * tf.square(self.output_i)
            loss_self_unbiased_i_masked = loss_self_unbiased_i_pointwise * tf.cast(1-self.ignoreTestUsers_mask, tf.float32)
            self.loss_self_unbiased_i = tf.reduce_sum(loss_self_unbiased_i_masked)

            loss_i_u_pos_rel_i_pointwise = self.input_R_i * tf.square(self.iAE_input_i - self.output_i)
            loss_i_u_pos_rel_i_masked = loss_i_u_pos_rel_i_pointwise * tf.cast(1-self.ignoreTestUsers_mask, tf.float32)
            self.loss_i_u_pos_rel_i = tf.reduce_sum( loss_i_u_pos_rel_i_masked  )
            
            self.reg_loss_i = self.reg_i*l2_loss(self.weights_i['encoder'], self.weights_i['decoder'],
                                             self.biases_i['encoder'], self.biases_i['decoder'])

            self.loss_i = self.reg_loss_i + self.loss_self_unbiased_i + self.w_i*self.loss_i_u_pos_rel_i


            # uAE update loss
            # ppscore_u = tf.clip_by_value(self.output_u, clip_value_min=0.1, clip_value_max = 1 )
            # self.loss_self_unbiased_u = tf.reduce_sum( self.input_R_u/ppscore_u * tf.square(1. - self.output_u) + (1 - self.input_R_u/ppscore_u) * tf.square(self.output_u) )
            # self.loss_i_u_pos_rel_u = tf.reduce_sum( self.input_R_u * tf.square(self.iAE_input_u - self.output_u)  )

            self.ignoreSetB_mask = tf.cast(self.ignoreSetB_mask, tf.int32)

            ppscore_u = tf.clip_by_value(self.output_u, clip_value_min=0.1, clip_value_max = 1 )
            loss_self_unbiased_u_pointwise = self.input_R_u/ppscore_u * tf.square(1. - self.output_u) + (1 - self.input_R_u/ppscore_u) * tf.square(self.output_u)
            loss_self_unbiased_u_masked = loss_self_unbiased_u_pointwise * tf.cast(1-self.ignoreSetB_mask, tf.float32)
            self.loss_self_unbiased_u = tf.reduce_sum(loss_self_unbiased_u_masked)

            loss_i_u_pos_rel_u_pointwise = self.input_R_u * tf.square(self.iAE_input_u - self.output_u)
            loss_i_u_pos_rel_u_masked = loss_i_u_pos_rel_u_pointwise * tf.cast(1-self.ignoreSetB_mask, tf.float32)
            self.loss_i_u_pos_rel_u = tf.reduce_sum( loss_i_u_pos_rel_u_masked  )

            self.reg_loss_u =  self.reg_u*l2_loss(self.weights_u['encoder'], self.weights_u['decoder'], 
                                             self.biases_u['encoder'], self.biases_u['decoder'])

            self.loss_u = self.reg_loss_u + self.loss_self_unbiased_u + self.w_u*self.loss_i_u_pos_rel_u  

        with tf.name_scope("val_loss"):
            # Mask the train loss for validation users and items from Set A
            # iAE update loss
            loss_self_unbiased_i_pointwise = self.val_target_R_i/ppscore_i * tf.square(1 - self.output_i) + (1 - self.val_target_R_i/ppscore_i) * tf.square(self.output_i)
            loss_self_unbiased_i_masked = loss_self_unbiased_i_pointwise * tf.cast(1-self.ignoreTestUsers_mask, tf.float32)
            self.val_loss_self_unbiased_i = tf.reduce_sum(loss_self_unbiased_i_masked)

            loss_i_u_pos_rel_i_pointwise = self.val_target_R_i * tf.square(self.iAE_input_i - self.output_i)
            loss_i_u_pos_rel_i_masked = loss_i_u_pos_rel_i_pointwise * tf.cast(1-self.ignoreTestUsers_mask, tf.float32)
            self.val_loss_i_u_pos_rel_i = tf.reduce_sum(loss_i_u_pos_rel_i_masked)

            self.val_loss_i = self.val_loss_self_unbiased_i + self.w_i*self.val_loss_i_u_pos_rel_i # ignore regularization loss

            # uAE update loss
            loss_self_unbiased_u_pointwise = self.val_target_R_u/ppscore_u * tf.square(1. - self.output_u) + (1 - self.val_target_R_u/ppscore_u) * tf.square(self.output_u)
            loss_self_unbiased_u_masked = loss_self_unbiased_u_pointwise * tf.cast(1-self.ignoreSetB_mask, tf.float32)
            self.val_loss_self_unbiased_u = tf.reduce_sum(loss_self_unbiased_u_masked)

            loss_i_u_pos_rel_u_pointwise = self.val_target_R_u * tf.square(self.iAE_input_u - self.output_u)
            loss_i_u_pos_rel_u_masked = loss_i_u_pos_rel_u_pointwise * tf.cast(1-self.ignoreSetB_mask, tf.float32)
            self.val_loss_i_u_pos_rel_u = tf.reduce_sum(loss_i_u_pos_rel_u_masked)

            self.val_loss_u = self.val_loss_self_unbiased_u + self.w_u*self.val_loss_i_u_pos_rel_u # ignore regularization loss


    def add_optimizer(self):
        with tf.name_scope("optimizer"):
            self.apply_grads_i = tf.compat.v1.train.AdagradOptimizer(learning_rate=self.eta_i).minimize(self.loss_i)
            self.apply_grads_u = tf.compat.v1.train.AdagradOptimizer(learning_rate=self.eta_u).minimize(self.loss_u)

    # Validation loss
    def get_valLoss(self, test_users, all_tr_u, all_tr_i, weights_enc_u, bias_enc_u, weights_dec_u, bias_dec_u, weights_enc_i, bias_enc_i, weights_dec_i, bias_dec_i):
        # uAE
        
        is_SetA = [True if item in range(50, 100) else False for item in all_tr_i]
        train_users = [user for user in range(len(all_tr_u)) if user not in test_users]
        is_trainUser = [True if user in train_users else False for user in range(len(all_tr_u))]
        
        uAE_ui_mat = sigmoid( np.matmul( sigmoid(np.matmul(self.train_ui_matrix, weights_enc_u) + bias_enc_u), weights_dec_u) + bias_dec_u)
        uAE_ui_mat = uAE_ui_mat.T
        iAE_iu_mat = sigmoid( np.matmul( sigmoid(np.matmul(self.train_iu_matrix, weights_enc_i) + bias_enc_i), weights_dec_i) + bias_dec_i)
        iAE_iu_mat = iAE_iu_mat.T

        # iAE
        val_loss_i = 0
        batch_num = int(len(all_tr_i) / self.batch_size_i) +1
        for b in range(batch_num):
            batch_set_idx = all_tr_i[b*self.batch_size_i : (b+1)*self.batch_size_i]
                
            batch_ignoreSetA = is_SetA[b*self.batch_size_i : (b+1)*self.batch_size_i] # Do not evaluate for items from SetA
            batch_mask = np.tile(np.expand_dims(batch_ignoreSetA, -1), [1,self.num_users]) 
            batch_mask[:, is_trainUser] = True # Do not evaluate for train Users

            batch_matrix = np.zeros((len(batch_set_idx), self.num_users))
            batch_matrix_val = np.zeros((len(batch_set_idx), self.num_users))
            uAE_bat_mat = np.zeros((len(batch_set_idx), self.num_users))
            for idx, item_id in enumerate(batch_set_idx):
                batch_matrix[idx] = self.train_iu_matrix[item_id]
                batch_matrix_val[idx] = self.val_iu_matrix[item_id]
                uAE_bat_mat[idx] = uAE_ui_mat[item_id]

            # pre-training only self bias
            feed_dict = {
                self.input_R_i: batch_matrix,
                self.val_target_R_i: batch_matrix_val,
                self.iAE_input_i: uAE_bat_mat,
                self.w_i: self.wi,
                self.ignoreTestUsers_mask : batch_mask
                }

            val_loss_i += self.sess.run([self.val_loss_i], feed_dict=feed_dict)[0]

        val_loss_u = 0
        batch_num = int(len(all_tr_u) / self.batch_size_u) +1
        for b in range(batch_num):
            batch_set_idx = all_tr_u[b*self.batch_size_u : (b+1)*self.batch_size_u]
                
            batch_ignoreTrainUsers = is_trainUser[b*self.batch_size_u : (b+1)*self.batch_size_u] # Do not evaluate for items from SetA
            batch_mask = np.tile(np.expand_dims(batch_ignoreTrainUsers, -1), [1,self.num_items]) 
            batch_mask[:, range(50, 100)] = True # Do not evaluate for Set A

            batch_matrix = np.zeros((len(batch_set_idx), self.num_items))
            batch_matrix_val = np.zeros((len(batch_set_idx), self.num_items))
            iAE_bat_mat = np.zeros((len(batch_set_idx), self.num_items))

            for idx, user_id in enumerate(batch_set_idx):
                batch_matrix[idx] = self.train_ui_matrix[user_id]
                batch_matrix_val[idx] = self.val_ui_matrix[user_id]
                iAE_bat_mat[idx] = iAE_iu_mat[user_id]

            # pre-training only self bias
            feed_dict = {
                self.input_R_u: batch_matrix,
                self.val_target_R_u: batch_matrix_val,
                self.iAE_input_u: iAE_bat_mat,
                self.w_u: self.wu,
                self.ignoreSetB_mask : batch_mask
                }

            val_loss_u += self.sess.run([self.val_loss_u], feed_dict=feed_dict)[0]

        return val_loss_i + val_loss_u

    def train_model(self, pscore, unbiased_eval, n_early_stop): # pscore and unbiased_eval were used for evaluation, but we evaluate on the stated preference ranking, so we do not need them
        init_op = tf.compat.v1.global_variables_initializer()
        self.sess.run(init_op)

        # max_score = 0
        min_loss = 1e20
        er_stop_count = 0
        er_stop_flag = False
        early_stop = n_early_stop

        all_tr_i = np.arange(self.num_items)
        all_tr_u = np.arange(self.num_users)

        train_losses = []
        val_losses = []

        for epoch in range(self.num_epochs):
            weights_enc_u, weights_dec_u, bias_enc_u, bias_dec_u = \
                    self.sess.run([self.weights_u['encoder'], self.weights_u['decoder'], self.biases_u['encoder'], self.biases_u['decoder']])

            uAE_ui_mat = sigmoid( np.matmul( sigmoid(np.matmul(self.train_ui_matrix, weights_enc_u) + bias_enc_u), weights_dec_u) + bias_dec_u)
            uAE_ui_mat = uAE_ui_mat.T
            
            weights_enc_i, weights_dec_i, bias_enc_i, bias_dec_i = \
                    self.sess.run([self.weights_i['encoder'], self.weights_i['decoder'], self.biases_i['encoder'], self.biases_i['decoder']])
            iAE_iu_mat = sigmoid( np.matmul( sigmoid(np.matmul(self.train_iu_matrix, weights_enc_i) + bias_enc_i), weights_dec_i) + bias_dec_i)
            iAE_iu_mat = iAE_iu_mat.T

            np.random.RandomState(12345).shuffle(all_tr_i)
            np.random.RandomState(12345).shuffle(all_tr_u)

            # For masking the loss for validation users and items from Set A/B
            ignoreSetB = [True if sum(self.train_ui_matrix[user, range(50)])==0 else False for user in all_tr_u]
            ignoreTestUsers = [True if item in range(50) else False for item in all_tr_i]
            test_users = [user for user in range(len(all_tr_u))if sum(self.train_ui_matrix[user, range(50)])==0]
            is_trainUser = [True if user not in test_users else False for user in range(len(all_tr_u))]

            # iAE
            train_loss_i = 0
            batch_num = int(len(all_tr_i) / self.batch_size_i) +1
            for b in range(batch_num):
                batch_set_idx = all_tr_i[b*self.batch_size_i : (b+1)*self.batch_size_i]
                
                batch_ignoreTestUsers = ignoreTestUsers[b*self.batch_size_i : (b+1)*self.batch_size_i]
                batch_ignoreTestUsers_mask = np.tile(np.expand_dims(batch_ignoreTestUsers, -1), [1,self.num_users])
                batch_ignoreTestUsers_mask[:, is_trainUser] = False

                batch_matrix = np.zeros((len(batch_set_idx), self.num_users))
                uAE_bat_mat = np.zeros((len(batch_set_idx), self.num_users))
                for idx, item_id in enumerate(batch_set_idx):
                    batch_matrix[idx] = self.train_iu_matrix[item_id]
                    uAE_bat_mat[idx] = uAE_ui_mat[item_id]

                # pre-training only self bias
                feed_dict = {
                    self.input_R_i: batch_matrix,
                    self.iAE_input_i: uAE_bat_mat,
                    self.w_i: self.wi,
                    self.ignoreTestUsers_mask : batch_ignoreTestUsers_mask
                    }

                _, loss_i = self.sess.run([self.apply_grads_i, self.loss_i], feed_dict=feed_dict)
                train_loss_i += loss_i

            # uAE
            train_loss_u = 0
            batch_num = int(len(all_tr_u) / self.batch_size_u) +1  # int(len(train_only_users) / self.batch_size_u) +1
            for b in range(batch_num):
                batch_set_idx = all_tr_u[b*self.batch_size_u : (b+1)*self.batch_size_u] # train_only_users[b*self.batch_size_u : (b+1)*self.batch_size_u]
                
                batch_ignoreSetB = ignoreSetB[b*self.batch_size_u : (b+1)*self.batch_size_u]
                batch_ignoreSetB_mask = np.tile(np.expand_dims(batch_ignoreSetB, -1), [1,100])
                batch_ignoreSetB_mask[:, range(50, 100)] = False

                batch_matrix = np.zeros((len(batch_set_idx), self.num_items))
                iAE_bat_mat = np.zeros((len(batch_set_idx), self.num_items))

                for idx, user_id in enumerate(batch_set_idx):
                    batch_matrix[idx] = self.train_ui_matrix[user_id]
                    iAE_bat_mat[idx] = iAE_iu_mat[user_id]
    
                # pre-training only self bias
                feed_dict = {
                    self.input_R_u: batch_matrix,
                    self.iAE_input_u: iAE_bat_mat,
                    self.w_u: self.wu,
                    self.ignoreSetB_mask : batch_ignoreSetB_mask
                    }
                
                
                _, loss_u = self.sess.run([self.apply_grads_u, 
                                           self.loss_u], 
                                           feed_dict=feed_dict)


                train_loss_u += loss_u
            train_losses.append(train_loss_i + train_loss_u)

            if epoch % 1 == 0 and not er_stop_flag and not self.val is None:
                weights_enc_u, weights_dec_u, weights_enc_i, weights_dec_i, bias_enc_u, bias_dec_u, bias_enc_i, bias_dec_i = \
                    self.sess.run([self.weights_u['encoder'], self.weights_u['decoder'], self.weights_i['encoder'], self.weights_i['decoder'], \
                                self.biases_u['encoder'], self.biases_u['decoder'], self.biases_i['encoder'], self.biases_i['decoder']])
                

                # validation
                # val_ret = unbiased_evaluator(user_embed=[weights_enc_u, weights_dec_u, weights_enc_i, weights_dec_i], item_embed=[bias_enc_u, bias_dec_u, bias_enc_i, bias_dec_i],
                #                             train=self.train, val=self.val, test=self.val, num_users=self.num_users, num_items=self.num_items, 
                #                             pscore=pscore, model_name=self.model_name, at_k=[3, 10], flag_test=False, flag_unbiased=True)
                val_loss = self.get_valLoss(test_users, all_tr_u, all_tr_i, weights_enc_u, bias_enc_u, weights_dec_u, bias_dec_u, weights_enc_i, bias_enc_i, weights_dec_i, bias_dec_i)
                if VERBOSE:
                    print(f"Validation loss: {val_loss:.4f}")
                val_losses.append(val_loss)

                # dim = self.hidden_dim_u
                # val_rets.append(val_ret.loc['Recall@10', f'proposed_{dim}'])

                if min_loss > val_loss: # max_score < val_ret.loc['Recall@10', f'proposed_{dim}']:
                    # max_score = val_ret.loc['Recall@10', f'proposed_{dim}']
                    # print("best_val_Recall@10: ", max_score)
                    min_loss = val_loss
                    if VERBOSE:
                        print("best validation loss: ", val_loss)
                    er_stop_count = 0 

                    self.best_weights_enc_u = weights_enc_u
                    self.best_weights_dec_u = weights_dec_u
                    self.best_bias_enc_u = bias_enc_u
                    self.best_bias_dec_u = bias_dec_u
                    self.best_weights_enc_i = weights_enc_i
                    self.best_weights_dec_i = weights_dec_i
                    self.best_bias_enc_i = bias_enc_i
                    self.best_bias_dec_i = bias_dec_i
                else:
                    er_stop_count += 1
                    if er_stop_count > early_stop:
                        if VERBOSE:
                            print("stopped!")
                        er_stop_flag = True
            if self.val is None:
                    self.best_weights_enc_u = weights_enc_u
                    self.best_weights_dec_u = weights_dec_u
                    self.best_bias_enc_u = bias_enc_u
                    self.best_bias_dec_u = bias_dec_u
                    self.best_weights_enc_i = weights_enc_i
                    self.best_weights_dec_i = weights_dec_i
                    self.best_bias_enc_i = bias_enc_i
                    self.best_bias_dec_i = bias_dec_i
            if er_stop_flag:
                break
            else:
                if VERBOSE:
                    print(epoch," uAE loss: %f   iAE loss: %f"%(train_loss_u, train_loss_i))
                    if self.val is not None:
                        print(epoch," validation loss: %f"%(val_loss))# val_ret.loc['Recall@10', f'proposed_{dim}']))
                    print("---------------------------------------------------------------------")

        self.sess.close()
        return train_losses, val_losses
