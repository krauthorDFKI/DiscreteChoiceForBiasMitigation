import tensorflow as tf
import numpy as np
# from utils.data import batch_iterator, BPRSampler
from Wan_2022_CPR.utils.graph import create_norm_adj, create_lightgcn_embed # , create_ngcf_embed
from Wan_2022_CPR.utils.inference import inner_product #, mlp
from Wan_2022_CPR.utils.loss import bpr_loss, l2_embed_loss
# from utils.tf_utils import init_variables, save_model
# from utils.evaluator import create_evaluators
# from utils.early_stopping import EarlyStopping
# from Lee_2022_BISER.models.cpr_paper.tools.monitor import Timer
# from Lee_2022_BISER.models.cpr_paper.tools.io import print_seperate_line
from Wan_2022_CPR.utils.dataset import Dataset

class BPR(object):
    def __init__(self, train, num_users: np.array, num_items: np.array, dim: int, lam: float, eta: float, n_i_group: int=1, n_u_group: int=1, ps_pow: float=None, clip_min: float=None, seed=0):
        # tf.compat.v1.set_random_seed(seed)

        self.lr = eta
        self.embed_size = dim
        self.reg = lam
        self.n_user = num_users
        self.n_item = num_items
        self.ps_pow = ps_pow # UBPR
        self.clip_min = clip_min # UBPR

        # self.timer = Timer()
        self.dataset = Dataset(train, n_i_group=n_i_group, n_u_group=n_u_group) # dataset

        # self.sess = tf.Session(
        #     config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        # )

        self._build_graph()
        # self.saver = tf.train.Saver(max_to_keep=1)

    def _build_graph(self):
        self.initializer = tf.initializers.GlorotUniform() # Previously in TF 1.X: tf.contrib.layers.xavier_initializer()
        self.create_variables()
        self.create_embeds()
        self.create_batch_ratings()
        # self.create_sampler()
        self.create_loss()
        self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss) # tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    # def fit(self, args, model_dir):

    #     init_variables(self.sess, self.saver, args.load_model, model_dir)

    #     # Create evaluators and early_stopping.
    #     self.evaluators = {}
    #     if args.eval_epoch is not None:
    #         self.evaluators = create_evaluators(
    #             self.dataset, args.eval_types, args.metrics, args.ks, args.n_thread
    #         )
    #         self.early_stopping = EarlyStopping(args.early_stop)

    #     # Start training and evaluation.
    #     # print_seperate_line()
    #     if args.eval_epoch is not None:
    #         self.eval(args)
    #         # print_seperate_line()

    #     for epoch in range(1, args.epoch + 1):
    #         self.train_1_epoch(epoch)

    #         if args.eval_epoch is not None and epoch % args.eval_epoch == 0:
    #             # print_seperate_line()
    #             self.eval(args)
    #             # print_seperate_line()

    #             if self.early_stopping.check_stop(self.evaluators, epoch):
    #                 break

    #     print(self.early_stopping)
    #     # print_seperate_line()

    #     # Save model.
    #     if args.save_model:
    #         save_model(self.sess, self.saver, args.verbose_name, args.epoch, model_dir)

    def create_variables(self):
        self.all_embeds_0 = tf.compat.v1.get_variable(
            "all_embeds_0",
            shape=[self.n_user + self.n_item, self.embed_size],
            initializer=self.initializer,
        )
        self.u_embeds_0, self.i_embeds_0 = tf.split(
            self.all_embeds_0, [self.n_user, self.n_item], 0
        )
        self.item_bias = tf.compat.v1.get_variable(
            "item_bias",
            shape=[self.n_item, 1],
            initializer=tf.compat.v1.zeros_initializer,
        )

        # if args.embed_type == "ngcf":
        #     self.W1s = []
        #     self.b1s = []
        #     self.W2s = []
        #     self.b2s = []
        #     for i in range(args.n_layer):
        #         self.W1s.append(
        #             tf.compat.v1.get_variable(
        #                 "W1_{}".format(i),
        #                 shape=[args.embed_size, self.embed_size],
        #                 initializer=self.initializer,
        #             )
        #         )
        #         self.b1s.append(
        #             tf.compat.v1.get_variable(
        #                 "b1_{}".format(i),
        #                 shape=[1, args.embed_size],
        #                 initializer=self.initializer,
        #             )
        #         )
        #         self.W2s.append(
        #             tf.compat.v1.get_variable(
        #                 "W2_{}".format(i),
        #                 shape=[args.embed_size, args.embed_size],
        #                 initializer=self.initializer,
        #             )
        #         )
        #         self.b2s.append(
        #             tf.compat.v1.get_variable(
        #                 "b2_{}".format(i),
        #                 shape=[1, args.embed_size],
        #                 initializer=self.initializer,
        #             )
        #         )

        # if args.inference_type == "mlp":
        #     weight_sizes = [2 * args.embed_size] + args.weight_sizes
        #     self.Ws = []
        #     self.bs = []
        #     for i in range(len(args.weight_sizes)):
        #         self.Ws.append(
        #             tf.compat.v1.get_variable(
        #                 "W_{}".format(i),
        #                 shape=[weight_sizes[i], weight_sizes[i + 1]],
        #                 initializer=self.initializer,
        #             )
        #         )
        #         self.bs.append(
        #             tf.compat.v1.get_variable(
        #                 "b_{}".format(i),
        #                 shape=[1, weight_sizes[i + 1]],
        #                 initializer=self.initializer,
        #             )
        #         )
        #     self.h = tf.compat.v1.get_variable(
        #         "h",
        #         shape=[weight_sizes[-1] + args.embed_size, 1],
        #         initializer=self.initializer,
        #     )

    def create_embeds(self):
        s_norm_adj = create_norm_adj(
            self.dataset.u_interacts,
            self.dataset.i_interacts,
            self.n_user,
            self.n_item,
        )

        # if args.embed_type == "ngcf":
        #     self.all_embeds = create_ngcf_embed(
        #         self.all_embeds_0,
        #         s_norm_adj,
        #         args.n_layer,
        #         self.W1s,
        #         self.b1s,
        #         self.W2s,
        #         self.b2s,
        #         args,
        #     )
        # elif args.embed_type == "lightgcn":
        # The following lines do not actually do anything here since n_layers is 0
        self.all_embeds = create_lightgcn_embed( # Zero layer lightgcn is equivalent to MF
            self.all_embeds_0, s_norm_adj, 0 #args.n_layer
        )

        self.u_embeds, self.i_embeds = tf.split(
            self.all_embeds, [self.n_user, self.n_item], 0
        )

    def create_loss(self):
        self.create_mf_loss()
        self.create_reg_loss()
        self.loss = self.mf_loss + self.reg_loss

    def create_mf_loss(self):
        self.batch_pos_i = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        self.batch_neg_i = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        batch_pos_i_embeds = tf.nn.embedding_lookup(self.i_embeds, self.batch_pos_i)
        batch_pos_i_bias = tf.nn.embedding_lookup(self.item_bias, self.batch_pos_i)
        batch_neg_i_embeds = tf.nn.embedding_lookup(self.i_embeds, self.batch_neg_i)
        batch_neg_i_bias = tf.nn.embedding_lookup(self.item_bias, self.batch_neg_i)
        # if args.inference_type == "inner_product":
        pos_scores = inner_product(self.batch_u_embeds, batch_pos_i_embeds) + batch_pos_i_bias
        neg_scores = inner_product(self.batch_u_embeds, batch_neg_i_embeds) + batch_neg_i_bias
        # elif args.inference_type == "mlp":
        #     pos_scores = mlp(
        #         self.batch_u_embeds, batch_pos_i_embeds, self.Ws, self.bs, self.h, args
        #     )
        #     neg_scores = mlp(
        #         self.batch_u_embeds, batch_neg_i_embeds, self.Ws, self.bs, self.h, args
        #     )
        self.mf_loss = bpr_loss(pos_scores, neg_scores)

    def create_reg_loss(self):
        self.reg_loss = self.reg * l2_embed_loss(self.all_embeds_0)
        # if args.embed_type == "ngcf":
        #     for x in self.W1s + self.b1s + self.W2s + self.b2s:
        #         self.reg_loss += args.weight_reg * tf.nn.l2_loss(x)
        # if args.inference_type == "mlp":
        #     for x in self.Ws + self.bs + [self.h]:
        #         self.reg_loss += args.weight_reg * tf.nn.l2_loss(x)

    def create_batch_ratings(self):
        self.batch_u = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        self.batch_u_embeds = tf.nn.embedding_lookup(self.u_embeds, self.batch_u)
        # # if args.inference_type == "inner_product":
        # self.batch_ratings = tf.matmul(
        #     self.batch_u_embeds, self.i_embeds, transpose_b=True
        # )
        # elif args.inference_type == "mlp":
        #     u_size = tf.shape(self.batch_u_embeds)[0]
        #     i_size = tf.shape(self.i_embeds)[0]
        #     u_repeats = tf.repeat(self.batch_u_embeds, i_size, axis=0)
        #     i_tiles = tf.tile(self.i_embeds, [u_size, 1])
        #     scores = mlp(u_repeats, i_tiles, self.Ws, self.bs, self.h, args)
        #     self.batch_ratings = tf.reshape(scores, [u_size, i_size])

    # def create_sampler(self, dataset, args):
    #     self.sampler = BPRSampler(dataset, args)

    # def train_1_epoch(self, epoch):
    #     self.timer.start("Epoch {}".format(epoch))
    #     losses = []
    #     mf_losses = []
    #     reg_losses = []
    #     for users, pos_items, neg_items in self.sampler.sample():
    #         _, batch_loss, batch_mf_loss, batch_reg_loss = self.sess.run(
    #             [self.opt, self.loss, self.mf_loss, self.reg_loss],
    #             feed_dict={
    #                 self.batch_u: users,
    #                 self.batch_pos_i: pos_items,
    #                 self.batch_neg_i: neg_items,
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

    # def eval(self, args):
    #     self.timer.start("Evaluation")
    #     for evaluator in self.evaluators.values():
    #         for idx, batch_u in enumerate(
    #             batch_iterator(evaluator.eval_users, args.eval_batch_size)
    #         ):
    #             batch_users_idx = range(
    #                 idx * args.eval_batch_size,
    #                 idx * args.eval_batch_size + len(batch_u),
    #             )
    #             batch_ratings = self.sess.run(
    #                 self.batch_ratings, feed_dict={self.batch_u: batch_u}
    #             )

    #             for idx, user in enumerate(batch_u):
    #                 batch_ratings[idx][self.dataset.train[user]] = -np.inf

    #             evaluator.update(batch_ratings, batch_users_idx)

    #         evaluator.update_final()
    #         print(evaluator)

    #     self.timer.stop()
