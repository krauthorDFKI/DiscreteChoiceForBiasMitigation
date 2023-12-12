
import numpy as np
import random as rd
import tensorflow as tf

from Wan_2022_CPR.bpr import BPR

from Lee_2022_BISER.custom_utils import tocsr, csr_to_user_dict

from src.config import VERBOSE

def cpr_pointwise_trainer(sess: tf.compat.v1.Session, data: str, model: BPR,
                      train: np.ndarray, val: np.ndarray, test: np.ndarray, num_users, num_items, 
                      pscore: np.ndarray, item_freq: np.ndarray, unbiased_eval: bool,
                      max_iters: int = 1000, batch_size: int = 2**12, 
                      model_name: str = 'relmf', date_now: str = '1', neg_sample: int = 3,
                      n_early_stop: int = 5):
    """Train and Evaluate Implicit Recommender."""

    # initialise all the TF variables
    init_op = tf.compat.v1.global_variables_initializer()
    sess.run(init_op)

    pscore = pscore[train[:, 1].astype(np.int32)]

    # train the given implicit recommender
    # max_score = 0
    min_val_loss = 1e20
    er_stop_count = 0
    best_u_emb =[]
    best_i_emb =[]
    early_stop = n_early_stop
    
    all_tr = np.arange(len(train))

    batch_size = batch_size

    train_losses = []

    tr_pos_index = np.where(train[:,2] > 0.5)[0]
    train_pos = train[tr_pos_index]
    train_dict = csr_to_user_dict(tocsr(train_pos, num_users, num_items))

    if val is not None:
        val_losses = []
        val_pos_index = np.where(val[:,2] > 0.5)[0]
        val_pos = val[val_pos_index]
        val_dict = csr_to_user_dict(tocsr(val_pos, num_users, num_items))
        val_users = [user_id for user_id, pos_items in val_dict.items() if len(pos_items)>0]

    if type(model).__name__ == "PD":        
        m = np.ones(num_items) # running popularity
        m = m / np.sum(m)

    for i in np.arange(max_iters):
        train_loss = 0

        users = []
        pos_items = []
        neg_items = []

        # neg sampling 이 아래 바꿔야함    
        if type(model).__name__ == 'CPR':
            pass
        else:
            # rd.seed(int(i))
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

        if type(model).__name__ != 'CPR':
            all_tr = np.arange(len(users))
        
            # shuffle 이 위에 바꿔야함
            # np.random.RandomState(12345).shuffle(all_tr)
            np.random.shuffle(all_tr)
            
            batch_num = int(len(all_tr) / batch_size) +1
            users = np.asarray(users)
            pos_items = np.asarray(pos_items)
            neg_items = np.asarray(neg_items)

            for b in range(batch_num):
                # mini-batch samples
                users_batch = users[all_tr[b*batch_size : (b+1)*batch_size]]
                pos_items_batch = pos_items[all_tr[b*batch_size : (b+1)*batch_size]]
                neg_items_batch = neg_items[all_tr[b*batch_size : (b+1)*batch_size]]

                if type(model).__name__ == 'PD': # update popularity
                    ids, counts = np.unique(pos_items_batch, return_counts=True)
                    update = np.asarray([counts[ids == idx][0] if idx in ids else 0 for idx in range(100)]).astype(float) # Handle items that were not picked
                    # update /= np.sum(update)

                    # Treat items from set A and B separately because of the different exposure
                    update_A = (update[50:] / update[50:].sum()) if update[50:].sum() > 0 else np.zeros(50) # update[50:].sum() == 0 can happen in the last batch
                    update_B = (update[:50] / update[:50].sum()) if update[:50].sum() > 0 else np.zeros(50) # update[:50].sum() == 0 can happen in the last batch
                    update = np.concatenate((update_B, update_A)) / 2 
                    # This way the most popular item from each set has the highest score among that set. This is also similar to how we calculate the propensity scores.
                    # The reason behind this lies in equation (5) of the original paper https://dl.acm.org/doi/pdf/10.1145/3404835.3462875:
                    # The parameter gamma treats high-popularity items more favorably than low-popularity items.
                    # Therefore, we treat items from set A and B separately to account for the different exposure.

                    m = model.alpha*m + (1-model.alpha)*update

                    m_powered = np.power(m, model.gamma)

                # update user-item latent factors and calculate training loss
                if type(model).__name__ == 'PD':
                    pos_pop_batch = np.asarray(m_powered[pos_items_batch], dtype=np.float32)
                    neg_pop_batch = np.asarray(m_powered[neg_items_batch], dtype=np.float32)
                    _, loss = sess.run([model.opt_pop_global, model.mf_loss_pop_global],
                                        feed_dict={
                                            model.users: users_batch,
                                            model.pos_items: pos_items_batch,
                                            model.neg_items: neg_items_batch,
                                            model.pos_pop: pos_pop_batch,
                                            model.neg_pop: neg_pop_batch,
                                        })
                else:
                    _, loss = sess.run([model.opt, model.mf_loss],
                                        feed_dict={
                                            model.batch_u: users_batch,
                                            model.batch_pos_i: pos_items_batch,
                                            model.batch_neg_i: neg_items_batch,
                                        })

                train_loss += loss
            train_loss /= batch_num
        else:
                losses = []
                for _ in range(model.sampler.n_step):
                    users, items = model.sampler.sample_batch(training=True)
                    _, loss = sess.run([model.opt, model.mf_loss_true],
                                                feed_dict={
                                                    model.batch_u: [l for ll in users for l in ll.flatten()], 
                                                    model.batch_pos_i: [l for ll in items for l in ll.flatten()]},
                                            )
                    losses.append(loss)
                train_loss += np.mean(losses)

        train_losses.append(train_loss)
        ############### evaluation
        if val is not None:
            if i % 1 == 0:
                # validation
                # at_k = 3
                # val_ret = unbiased_evaluator(user_embed=u_emb, item_embed=i_emb, 
                #                         train=train, val=val, test=val, num_users=num_users, num_items=num_items, 
                #                         pscore=item_freq, model_name=model_name, at_k=[at_k], flag_test=False, flag_unbiased = True)
                
                # neg sampling
                users_val = []
                pos_items_val = []
                neg_items_val = []
                masks_val = []

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
                
                
                if type(model).__name__ == 'CPR':
                    losses = []
                    for _ in range(model.sampler.n_step):
                        users, items = model.sampler.sample_batch(training=False) # CPR considers multiple batches of differently shaped data
                        loss_batch, u_emb, i_emb, i_bias = sess.run([model.mf_loss_true, 
                                                                model.u_embeds, 
                                                                model.i_embeds,
                                                                model.item_bias],
                                                    feed_dict={
                                                        model.batch_u: [l for ll in users for l in ll.flatten()], 
                                                        model.batch_pos_i: [l for ll in items for l in ll.flatten()]},
                        )
                        losses.append(loss_batch)
                    val_loss = np.mean(losses)     
                elif type(model).__name__ == 'PD':
                    pos_pop_val = np.asarray(m_powered[pos_items_val], np.float32)
                    neg_pop_val = np.asarray(m_powered[neg_items_val], np.float32)
                    val_loss, u_emb, i_emb, i_bias = sess.run([model.mf_loss_pop_global, 
                                                                model.weights['user_embedding'], 
                                                                model.weights['item_embedding'],
                                                                model.weights['item_bias']
                                                                ],
                                                                feed_dict={
                                                                    model.users: users_val,
                                                                    model.pos_items: pos_items_val,
                                                                    model.neg_items: neg_items_val,
                                                                    model.pos_pop: pos_pop_val,
                                                                    model.neg_pop: neg_pop_val,
                                                                })
                else:
                    val_loss, u_emb, i_emb, i_bias = sess.run([model.mf_loss, 
                                                                model.u_embeds_0, 
                                                                model.i_embeds_0,
                                                                model.item_bias
                                                                ],
                                                                feed_dict = {model.batch_u: users_val,
                                                                            model.batch_pos_i: pos_items_val,
                                                                            model.batch_neg_i: neg_items_val})
                val_losses.append(val_loss)
                if VERBOSE:
                    print(f"{i}: Train Loss {train_loss:.4f},  Validation Loss {val_loss:.4f}",)

                # dim = u_emb.shape[1]
                # best_score = val_ret.loc[f'MAP@{at_k}', f'{model_name}_{dim}']

                if val_loss < min_val_loss: # max_score < best_score:
                    # max_score = best_score
                    # print(f"best_val_MAP@{at_k}: ", max_score)

                    min_val_loss = val_loss
                    er_stop_count = 0
                    
                    best_u_emb = u_emb
                    best_i_emb = [i_emb, i_bias]
                    
                    if VERBOSE:
                        print(f"best validation loss: {min_val_loss:.4f}")

                else:
                    er_stop_count += 1
                    if er_stop_count > early_stop:
                        if VERBOSE:
                            print("stopped!")
                        break
    if val is None or best_i_emb == []:               
        if type(model).__name__ == 'PD':
            u_emb, i_emb, i_bias = sess.run([model.weights['user_embedding'], 
                                            model.weights['item_embedding'],
                                            model.weights['item_bias']
                                            ])    
        elif type(model).__name__ == 'CPR':
            u_emb, i_emb, i_bias = sess.run([
                                        model.u_embeds, 
                                        model.i_embeds,
                                        model.item_bias
                                    ])
        else:
            u_emb, i_emb, i_bias = sess.run([model.u_embeds_0, model.i_embeds_0, model.item_bias])
        best_u_emb = u_emb
        best_i_emb = [i_emb, i_bias]
        
        if val is None:
            val_losses = None

    sess.close()

    return best_u_emb, best_i_emb, train_losses, val_losses