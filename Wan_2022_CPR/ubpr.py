import tensorflow as tf
import numpy as np
from Wan_2022_CPR.bpr import BPR
# from utils.data import UBPRSampler
from Wan_2022_CPR.utils.loss import ubpr_loss


class UBPR(BPR):
    # def create_sampler(self, dataset, args):
    #     self.sampler = UBPRSampler(dataset, args)

    def create_mf_loss(self):
        self.batch_pos_i = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        self.batch_neg_i = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        # self.batch_j_labels = tf.compat.v1.placeholder(tf.float32, shape=(None,))
        batch_i_embeds = tf.nn.embedding_lookup(self.i_embeds, self.batch_pos_i)
        batch_i_bias = tf.nn.embedding_lookup(self.item_bias, self.batch_pos_i)
        batch_j_embeds = tf.nn.embedding_lookup(self.i_embeds, self.batch_neg_i)
        batch_j_bias = tf.nn.embedding_lookup(self.item_bias, self.batch_neg_i)
        # self.p_scores = np.power(
        #     self.dataset.i_degrees / self.dataset.i_degrees.max(),
        #     self.ps_pow,
        #     dtype=np.float32,
        # )
        p_scores_A = np.power(
            self.dataset.i_degrees[50:] / self.dataset.i_degrees[50:].max(),
            self.ps_pow,
            dtype=np.float32,
        )
        p_scores_B = np.power(
            self.dataset.i_degrees[:50] / self.dataset.i_degrees[:50].max(),
            self.ps_pow,
            dtype=np.float32,
        )
        self.p_scores = np.append(p_scores_B, p_scores_A)
        batch_i_p_scores = tf.nn.embedding_lookup(self.p_scores, self.batch_pos_i)
        # batch_j_p_scores = tf.nn.embedding_lookup(self.p_scores, self.batch_j)
        self.mf_loss = ubpr_loss(
            self.batch_u_embeds,
            batch_i_embeds,
            batch_i_bias,
            batch_j_embeds,
            batch_j_bias,
            batch_i_p_scores,
            # batch_j_p_scores,
            # self.batch_j_labels,
            self.clip_min
        )

    # def train_1_epoch(self, epoch):
    #     self.timer.start("Epoch {}".format(epoch))
    #     losses = []
    #     mf_losses = []
    #     reg_losses = []
    #     for users, i_items, j_items, j_labels in self.sampler.sample():
    #         _, batch_loss, batch_mf_loss, batch_reg_loss = self.sess.run(
    #             [self.opt, self.loss, self.mf_loss, self.reg_loss],
    #             feed_dict={
    #                 self.batch_u: users,
    #                 self.batch_i: i_items,
    #                 self.batch_j: j_items,
    #                 self.batch_j_labels: j_labels,
    #             },
    #         )
    #         losses.append(batch_loss)
    #         mf_losses.append(batch_mf_loss)
    #         reg_losses.append(batch_reg_loss)
    #     self.timer.stop(
    #         "loss = {:.5f} = {:.5f} + {:.5f}".format(
    #             np.mean(losses), np.mean(mf_losses), np.mean(reg_losses)
    #         )
    #     )
