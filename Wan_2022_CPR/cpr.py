import tensorflow as tf
import numpy as np
from Wan_2022_CPR.bpr import BPR
from Wan_2022_CPR.utils.data import CPRSampler
from Wan_2022_CPR.utils.inference import inner_product, mlp
from Wan_2022_CPR.utils.loss import cpr_loss


class CPR(BPR):
    # def create_sampler(self, dataset, args):
        # self.sampler = CPRSampler(dataset, args)

    def __init__(self, train, num_users: np.array, num_items: np.array, dim: int, lam: float, eta: float, batch_size: int, 
                 batch_total_sample_sizes: int, beta: float, max_k_interact: int, gamma: float, n_step: int, val=None, seed=0):
        self.batch_size = batch_size
        self.beta = beta
        self.sampler = CPRSampler(train, val=val, max_k_interact=max_k_interact, gamma=gamma, 
                                  batch_size=batch_size, beta=beta, num_users=num_users, 
                                  num_items=num_items, neg_sample_rate=None, n_step=n_step)

        super().__init__(train, num_users, num_items, dim, lam, eta, n_i_group=1, n_u_group=1, ps_pow=1, clip_min=0, seed=seed)

    def create_mf_loss(self): # , args):
        # NO BIAS TERM HERE AS IT CANCELS OUT IN CPR LOSS
        self.batch_pos_i = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        batch_pos_i_embeds = tf.nn.embedding_lookup(self.i_embeds, self.batch_pos_i)
        pos_scores = []
        neg_scores = []
        # u_embeds: u1, u2, u1, u2, u3, ...
        u_splits = tf.split(
            self.batch_u_embeds, self.sampler.batch_total_sample_sizes, 0
        )
        i_splits = tf.split(
            batch_pos_i_embeds, self.sampler.batch_total_sample_sizes, 0
        )
        for idx in range(len(self.sampler.batch_total_sample_sizes)):
            u_list = tf.split(u_splits[idx], idx + 2, 0)
            i_list = tf.split(i_splits[idx], idx + 2, 0)
            # if args.inference_type == "inner_product":
            pos_scores.append(
                tf.reduce_mean(
                    [inner_product(u, i) for u, i in zip(u_list, i_list)], axis=0
                )
            )
            neg_scores.append(
                tf.reduce_mean(
                    [
                        inner_product(u, i)
                        for u, i in zip(u_list, i_list[1:] + [i_list[0]],)
                    ],
                    axis=0,
                )
            )
            # elif args.inference_type == "mlp":
            #     pos_scores.append(
            #         tf.reduce_mean(
            #             [
            #                 mlp(u, i, self.Ws, self.bs, self.h, args)
            #                 for u, i in zip(u_list, i_list)
            #             ],
            #             axis=0,
            #         )
            #     )
            #     neg_scores.append(
            #         tf.reduce_mean(
            #             [
            #                 mlp(u, i, self.Ws, self.bs, self.h, args)
            #                 for u, i in zip(u_list, i_list[1:] + [i_list[0]])
            #             ],
            #             axis=0,
            #         )
            #     )
        pos_scores = tf.concat(pos_scores, axis=0)
        neg_scores = tf.concat(neg_scores, axis=0)

        self.mf_loss, self.mf_loss_true = cpr_loss(pos_scores, neg_scores, self.beta, self.batch_size) # args)

    # def train_1_epoch(self, epoch):
    #     self.timer.start("Epoch {}".format(epoch))
    #     losses = []
    #     mf_losses = []
    #     reg_losses = []
    #     for users, items in self.sampler.sample():
    #         _, batch_loss, batch_mf_loss, batch_reg_loss = self.sess.run(
    #             [self.opt, self.loss, self.mf_loss, self.reg_loss],
    #             feed_dict={self.batch_u: users, self.batch_pos_i: items},
    #         )
    #         losses.append(batch_loss)
    #         mf_losses.append(batch_mf_loss)
    #         reg_losses.append(batch_reg_loss)
    #     self.timer.stop(
    #         "loss = {:.5f} = {:.5f} + {:.5f}".format(
    #             np.mean(losses), np.mean(mf_losses), np.mean(reg_losses)
    #         )
    #     )

from Wan_2022_CPR.dice import Sampler
# Adapted from the original implementation at https://github.com/Qcactus/CPR/blob/782edb82ea8e996071807a9e5c5f9ef1c477e6ce/recq/utils/data.py#L101
# Adapted according to the DICE sampler from the same Repo.
# Adapter according to our needs.
class CPRSampler(Sampler):
    def __init__(self, train, max_k_interact, gamma, batch_size, beta,
                 num_users, num_items, neg_sample_rate, n_step, val=None):
        super().__init__(train, num_users, num_items, neg_sample_rate, val)

        self.sample_rate = beta
        # self.batch_size = batch_size
        self.max_k_interact = max_k_interact
        # self.batch_total_sample_sizes = [2**k for k in range(2, max_k_interact)]
        self.n_step = n_step
        self.sample_ratio = gamma

        # HABE ROLLE VON SAMPLE RATE UND RAATIO VERTAUSCHT; SAMPLE RATIO IST HIER EGAL; MUSS IN LOSS FUNCTION DOCH TOTAL SAMPLE SIZES DINGS VERWENDEN

        self.set_samplers(k_interact=None,
                          sample_ratio=self.sample_ratio,
                          max_k_interact=self.max_k_interact,
                          batch_size=batch_size,
                          sample_rate=self.sample_rate)

    def set_samplers(self, k_interact, sample_ratio, max_k_interact, batch_size, sample_rate):
        if k_interact is None:
            ratios = np.power(
                sample_ratio, np.arange(max_k_interact - 2, -1, -1)
            )
        else:
            ratios = np.array([0] * (k_interact - 2) + [1])
        batch_sizes = np.round(batch_size / np.sum(ratios) * ratios).astype( # This line seems to aim at making all batches * beta add up to the given batch_size
            np.int32
        )
        batch_sizes[-1] = batch_size - np.sum(batch_sizes[:-1])
        self.batch_sample_sizes = np.ceil( # size of b*beta*gamma (line 2 from the algorithm in the original paper)
            np.array(batch_sizes) * sample_rate
        ).astype(np.int32)
        self.batch_total_sample_sizes = self.batch_sample_sizes * np.arange( # total number of user-item pairs to be sampled per batch (k times batch_sample_sizes)
            2, len(self.batch_sample_sizes) + 2
        )
        self.batch_sample_size = np.sum(self.batch_total_sample_sizes) # total number of user-item pairs to be samples
        self.sample_size = self.n_step * self.batch_sample_size
        self.batch_choice_sizes = 2 * self.batch_sample_sizes
        self.choice_size = 2 * self.sample_size

        self.users = np.empty(self.sample_size, dtype=np.int32)
        self.items = np.empty(self.sample_size, dtype=np.int32)
        # self.cpr_sampler = CyCPRSampler(
        #     self.train,
        #     self.u_interacts,
        #     self.i_interacts,
        #     self.users,
        #     self.items,
        #     self.n_step,
        #     self.batch_sample_sizes,
        #     args.n_thread,
        # )

    def sample_batch_k(self, k, training: bool):
        # We go with the latter as the former is ill-defined for BPR
        if training:
            record = self.record
            lil_record = self.lil_record
        else:
            record = self.record_val
            lil_record = self.lil_record_val

        # Algorithm 1: Dynamic Sampling from the original CPR Paper at https://dl.acm.org/doi/pdf/10.1145/3485447.3512010
        # Line 1: Initialize sample set S
        S = []
        # Line 2: Randomly select batch_size * self.beta * self.gamma samples, each of which contains k positive user-item pairs
        while len(S) < self.batch_sample_sizes[k-2]:
            n_samples_batch = self.batch_sample_sizes[k-2]
            # Draw user-item pairs
            random_indices = np.random.choice(a=range(len(record)), size=(n_samples_batch,k), replace=True)
            
            # Line 3 to 6: Append negative pairs to sample set S
            user_item_pairs = np.reshape(record[np.reshape(random_indices, -1)], (n_samples_batch, k, 2))
            # Check if this is a set of negative pairs
            legal_pairs = user_item_pairs[
                                [np.sum(
                                    [j in lil_record[i] for i,j in zip(user_item_pair[:, 0], list(user_item_pair[1:, 1]) + [user_item_pair[0, 1]])]
                                    ) == 0
                                for user_item_pair in user_item_pairs]
                            ]
            if training:
                legal_pairs = [
                    legal_pair for legal_pair in legal_pairs if
                        np.sum(
                            [item < 50 # Item from list B
                             and user in self.users_val # Validation user
                             for user, item in zip(legal_pair[:, 0], list(legal_pair[1:, 1]) + [legal_pair[0, 1]])]
                            ) == 0
                ]

            S.extend(list(
                legal_pairs[:min(len(legal_pairs), self.batch_sample_sizes[k-2]-len(S))]
                ))

        return np.asarray(S)

    def sample_batch(self, training: bool):
        batches_users = []
        batches_items = []
        for k in range(2, self.max_k_interact + 1):
            S = self.sample_batch_k(k, training)
            batches_users.append(S[:,:,0])
            batches_items.append(S[:,:,1])
        return batches_users, batches_items