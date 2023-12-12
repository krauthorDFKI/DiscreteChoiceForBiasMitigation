
import numpy as np
import random as rd
import tensorflow as tf

from scipy import sparse

from Lee_2022_BISER.models.cjmf import CJMF
from Lee_2022_BISER.models.recommenders import PointwiseRecommender
from Lee_2022_BISER.evaluate.evaluator import aoa_evaluator, unbiased_evaluator

from Lee_2022_BISER.custom_utils import sigmoid, tocsr, csr_to_user_dict, csr_to_user_dict_neg

from src.config import VERBOSE

def pointwise_trainer(sess: tf.compat.v1.Session, data: str, model: PointwiseRecommender,
                      train: np.ndarray, val: np.ndarray, test: np.ndarray, num_users, num_items, 
                      pscore: np.ndarray, item_freq: np.ndarray, unbiased_eval: bool,
                      max_iters: int = 1000, batch_size: int = 2**12, 
                      model_name: str = 'relmf', date_now: str = '1', n_early_stop: int = 20):
    """Train and Evaluate Implicit Recommender."""

    # initialise all the TF variables
    init_op = tf.compat.v1.global_variables_initializer()
    sess.run(init_op)

    if val is not None:
        pscore_val = pscore[val[:, 1].astype(np.int32)]
    pscore_train = pscore[train[:, 1].astype(np.int32)]

    # train the given implicit recommender
    # max_score = 0
    min_val_loss = 1e20
    er_stop_count = 0
    best_u_emb =[]
    best_i_emb =[]
    
    all_tr = np.arange(len(train))

    batch_size = batch_size
    early_stop = n_early_stop

    train_losses = []
    if val is not None:
        val_losses = []
    
    for i in np.arange(max_iters):
        # np.random.RandomState(12345).shuffle(all_tr)
        np.random.shuffle(all_tr)
        
        train_loss = 0

        batch_num = np.ceil(len(all_tr) / batch_size).astype(int) # int(len(all_tr) / batch_size) +1
        for b in range(batch_num):
            # mini-batch samples
            train_batch = train[all_tr[b*batch_size : (b+1)*batch_size]]
            train_label = train_batch[:, 2]
            train_score = pscore_train[all_tr[b*batch_size : (b+1)*batch_size]]

            # update user-item latent factors and calculate training loss
            _, loss = sess.run([model.apply_grads, model.weighted_mse],
                            feed_dict={model.users: train_batch[:, 0],
                                        model.items: train_batch[:, 1],
                                        model.labels: np.expand_dims(train_label, 1),
                                        model.scores: np.expand_dims(train_score, 1)
                                        })
            assert(not np.isnan(loss))
            train_loss += loss
        train_loss /= batch_num
        train_losses.append(train_loss)
        ############### evaluation
        if val is not None:
            if i % 1 == 0:
                # validation
                at_k = 3
                # val_ret = unbiased_evaluator(user_embed=u_emb, item_embed=i_emb, 
                #                         train=train, val=val, test=val, num_users=num_users, num_items=num_items, 
                #                         pscore=item_freq, model_name=model_name, at_k=[at_k], flag_test=False, flag_unbiased = True)
                val_loss, u_emb, i_emb, i_bias = sess.run([model.weighted_mse, model.user_embeddings, model.item_embeddings, model.item_bias],
                                    feed_dict={model.users: val[:, 0],
                                                model.items: val[:, 1],
                                                model.labels: np.expand_dims(val[:, 2], 1),
                                                model.scores: np.expand_dims(pscore_val, 1)
                                                })
                val_losses.append(val_loss)
                if VERBOSE:
                    print(f"{i}: Train Loss {train_loss:.4f},  Validation Loss {val_loss:.4f}",)

                # dim = u_emb.shape[1]
                # best_score = val_ret.loc[f'MAP@{at_k}', f'{model_name}_{dim}']

                if val_loss < min_val_loss: # max_score < best_score:
                    # max_score = best_score
                    # print(f"best_val_MAP@{at_k}: ", max_score)
                    min_val_loss = val_loss
                    if VERBOSE:
                        print(f"best validation loss: {min_val_loss:.4f}")
                    er_stop_count = 0
                    
                    best_u_emb = u_emb
                    best_i_emb = [i_emb, i_bias]

                else:
                    er_stop_count += 1
                    if er_stop_count > early_stop:
                        if VERBOSE:
                            print("stopped!")
                        break
    if val is None:                    
        u_emb, i_emb, i_bias = sess.run([model.user_embeddings, model.item_embeddings, model.item_bias])
        best_u_emb = u_emb
        best_i_emb = [i_emb, i_bias]
        val_losses = None

    sess.close()

    return best_u_emb, best_i_emb, train_losses, val_losses

def macr_trainer(sess: tf.compat.v1.Session, data: str, model: PointwiseRecommender,
                      train: np.ndarray, val: np.ndarray, test: np.ndarray, num_users, num_items, 
                      pscore: np.ndarray, item_freq: np.ndarray, unbiased_eval: bool,
                      max_iters: int = 1000, batch_size: int = 2**12, 
                      model_name: str = 'pd', date_now: str = '1', neg_sample: int = 30, n_early_stop: int = 20):
    """Train and Evaluate Implicit Recommender."""

    # initialise all the TF variables
    init_op = tf.compat.v1.global_variables_initializer()
    sess.run(init_op)

    # train the given implicit recommender
    # max_score = 0
    min_val_loss = 1e20
    er_stop_count = 0
    best_u_emb =[]
    best_i_emb =[]

    batch_size = batch_size
    early_stop = n_early_stop

    train_losses = []
    if val is not None:
        val_losses = []

    # Coat, Yahoo test set are made in advance
    tr_pos_index = np.where(train[:,2] > 0.5)[0]
    # if data in ['coat', 'yahoo', "DFKI"]:
    #     val_pos_index = np.where(val[:,2] > -1)[0]
    # else:    
    #     val_pos_index = np.where(val[:,2] > 0.5)[0]
    if val is not None:
        val_pos_index = np.where(val[:,2] > 0.5)[0]
    train_pos = train[tr_pos_index]
    if val is not None:
        val_pos = val[val_pos_index]
    # val_pos[:, 2] = 1

    if val is not None:
        train_val = np.r_[train_pos, val_pos]
    train_dict = csr_to_user_dict(tocsr(train_pos, num_users, num_items))
    if val is not None:
        val_dict = csr_to_user_dict(tocsr(val_pos, num_users, num_items))
        val_users = [user_id for user_id, pos_items in val_dict.items() if len(pos_items)>0]

    # itemlists = list(range(num_items))
    # itemlists = np.asarray(itemlists)

    for i in np.arange(max_iters):
        train_loss = 0

        users = []
        pos_items = []
        neg_items = []

        rd.seed(int(i))
        # neg sampling 이 아래 바꿔야함
        for user_idx in range(num_users):
            pos_items_one_user = train_dict[user_idx]
            pos_items_one_user = pos_items_one_user * neg_sample
            neg_items_one_user = []
            
            for interaction_idx in range(len(pos_items_one_user)):
                while True:
                    if pos_items_one_user[interaction_idx] < 50:
                        neg_item = rd.choice(range(50))
                    else:
                        neg_item = rd.choice(range(50, 100))
                    if neg_item not in pos_items_one_user:
                        neg_items_one_user.append(neg_item)
                        break

            user_one_user = [user_idx]*len(pos_items_one_user) 

            users.extend(user_one_user)
            pos_items.extend(pos_items_one_user)
            neg_items.extend(neg_items_one_user)

        all_tr = np.arange(len(users))
        
        # shuffle 이 위에 바꿔야함
        np.random.RandomState(12345).shuffle(all_tr)
        
        batch_num = int(len(all_tr) / batch_size) +1
        users = np.asarray(users)
        pos_items = np.asarray(pos_items)
        neg_items = np.asarray(neg_items)

        for b in range(batch_num):
            # mini-batch samples
            users_batch = users[all_tr[b*batch_size : (b+1)*batch_size]]
            pos_items_batch = pos_items[all_tr[b*batch_size : (b+1)*batch_size]]
            neg_items_batch = neg_items[all_tr[b*batch_size : (b+1)*batch_size]]

            _, loss = sess.run([model.opt_two_bce_both, model.loss_two_bce_both],
                            feed_dict = {model.users: users_batch,
                                        model.pos_items: pos_items_batch,
                                        model.neg_items: neg_items_batch})
            train_loss += loss
        train_loss /= batch_num
        train_losses.append(train_loss)

        ############### evaluation
        if val is not None:
            if i % 1 == 0:
                # print(i,":  ", loss)
                # u_emb_t, i_emb_t, w_user = sess.run(
                #     [model.weights['user_embedding'], model.weights['item_embedding'], model.w_user])
                
                # u_emb = [u_emb_t, i_emb_t, w_user]
                # i_emb = 0
                
                # # inference
                # batch_ratings = u_emb_t @ i_emb_t.T
                # user_scores = u_emb_t @ w_user
                # pred = batch_ratings*sigmoid(user_scores) 

                # validation
                # at_k = 3
                # val_ret = unbiased_evaluator(user_embed=u_emb, item_embed=i_emb, 
                #                             train=train, val=val, test=val, num_users=num_users, num_items=num_items, 
                #                             pscore=item_freq, model_name=model_name, at_k=[at_k], flag_test=False, \
                #                             flag_unbiased = True, pred=pred)
                users_val = []
                pos_items_val = []
                neg_items_val = []

                # neg sampling
                for user_idx in val_users: # in range(num_users)
                    pos_items_one_user = val_dict[user_idx]
                    pos_items_one_user = pos_items_one_user * neg_sample
                    neg_items_one_user = []

                    for _ in range(len(pos_items_one_user)):
                        while True:
                            neg_item = rd.choice(range(50))
                            if neg_item not in pos_items_one_user:
                                neg_items_one_user.append(neg_item)
                                break

                    user_one_user = [user_idx]*len(pos_items_one_user) 

                    users_val.extend(user_one_user)
                    pos_items_val.extend(pos_items_one_user)
                    neg_items_val.extend(neg_items_one_user)
                users_val = np.asarray(users_val)
                pos_items_val = np.asarray(pos_items_val)
                neg_items_val = np.asarray(neg_items_val)
                
                val_loss, u_emb_t, i_emb_t, i_bias, w_user, w_item = sess.run([model.mf_loss_two_bce_both, 
                                                            model.weights['user_embedding'], 
                                                            model.weights['item_embedding'],
                                                            model.weights['item_bias'], 
                                                            model.w_user,
                                                            model.w],
                                                            feed_dict = {model.users: users_val,
                                                                        model.pos_items: pos_items_val,
                                                                        model.neg_items: neg_items_val})
                u_emb = [u_emb_t, i_emb_t, i_bias, w_user, w_item, model.c]
                i_emb = 0
                val_losses.append(val_loss) 
                if VERBOSE:
                    print(f"{i}: Train Loss {train_loss:.4f},  Validation Loss {val_loss:.4f}",)
                

                # dim = model.emb_dim
                # best_score = val_ret.loc[f'MAP@{at_k}', f'{model_name}_{dim}']

                if min_val_loss > val_loss: # max_score < best_score:
                    # max_score = best_score
                    # print(f"best_val_MAP@{at_k}: ", max_score)
                    min_val_loss = val_loss
                    if VERBOSE:
                        print(f"best validaton loss: ", min_val_loss)
                    
                    er_stop_count = 0
                                    
                    best_u_emb = u_emb
                    best_i_emb = i_emb

                else:
                    er_stop_count += 1
                    if er_stop_count > early_stop:
                        if VERBOSE:
                            print("stopped!")
                        break
        else:
            if VERBOSE:
                print(f"{i}: Train Loss {train_loss:.4f}",)
    if val is None:                    
        u_emb_t, i_emb_t, i_bias, w_user, w_item  = sess.run([model.weights['user_embedding'], 
                                                    model.weights['item_embedding'],
                                                    model.weights['item_bias'], 
                                                    model.w_user,
                                                    model.w])
        u_emb = [u_emb_t, i_emb_t, i_bias, w_user, w_item, model.c]
        i_emb = 0
        best_u_emb = u_emb
        best_i_emb = i_emb
        val_losses = None
    sess.close()

    return best_u_emb, best_i_emb, train_losses, val_losses


def cjmf_trainer(sess: tf.compat.v1.Session, data: str, n_components: int, num_users: int, num_items: int, item_freq: np.ndarray, batch_size: int, max_iters: int,\
                 unbiased_eval: bool, C:int,  lr:float, reg:float, alpha:float, beta:float, train: np.ndarray, val: np.ndarray, seed: int, model_name: str):

    batch_size = batch_size
    early_stop = 5

    item_pop = item_freq**2/3
    item_pop = item_pop/np.max(item_pop)
    
    """Train CJMF models."""
    model = CJMF(sess=sess, num_item=num_items, num_user=num_users, hidden=n_components, batch_size=batch_size, data_name=data, epoch=max_iters, item_pop=item_pop, \
                C=C, lr=lr, alpha=alpha, beta=beta, train=train, val=val, early_stop=early_stop, unbiased_eval=unbiased_eval, reg=reg, item_freq=item_freq, seed=seed)

    best_P_list, best_Q_list = model.run()

    return best_P_list, best_Q_list