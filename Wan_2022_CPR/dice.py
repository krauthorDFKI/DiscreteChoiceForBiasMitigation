import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# from time import time
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
# from utils.data import batch_iterator, DICESampler
from Wan_2022_CPR.utils.graph import create_norm_adj, create_lightgcn_embed # , create_ngcf_embed
from Wan_2022_CPR.utils.inference import inner_product #, mlp
from Wan_2022_CPR.utils.loss import bpr_loss, mask_bpr_loss, l2_embed_loss, discrepency_loss
# from utils.tf_utils import init_variables, save_model
# from utils.evaluator import create_evaluators
# from utils.early_stopping import EarlyStopping
# from Lee_2022_BISER.models.cpr_paper.tools.monitor import Timer
# from Lee_2022_BISER.models.cpr_paper.tools.io import print_seperate_line
from Wan_2022_CPR.utils.dataset import Dataset


class DICE(object):
    """LightGCN model

    SIGIR 2020. He, Xiangnan, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang.
    "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." arXiv
    preprint arXiv:2002.02126 (2020).
    """

    def __init__(self, train, num_users: np.array, num_items: np.array, dim: int, lam: float, eta: float, 
                 dis_pen, init_int_weight: float, init_pop_weight: float, 
                 init_margin: float, pool: float, loss_decay: float, margin_decay: float, seed=0):
        """Initializing the model. Create parameters, placeholders, embeddings and loss function."""
        tf.compat.v1.set_random_seed(seed)

        self.lr = eta
        self.embed_size = dim
        self.reg = lam
        self.n_user = num_users
        self.n_item = num_items
        self.dis_pen = dis_pen
        self.init_int_weight = init_int_weight
        self.init_pop_weight = init_pop_weight
        self.init_margin = init_margin
        self.pool = pool
        self.loss_decay = loss_decay
        self.margin_decay = margin_decay

        # self.timer = Timer()
        self.dataset = Dataset(train, n_i_group=1, n_u_group=1) # dataset

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
        # self.sampler = DICESampler(self.dataset, args)
        self.int_weight = tf.compat.v1.placeholder(tf.float32, shape=())
        self.pop_weight = tf.compat.v1.placeholder(tf.float32, shape=())
        # self.dis_pen = dis_pen
        self.create_loss()
        self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss) # tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    # def fit(self, args, model_dir):

    #     init_variables(self.sess, self.saver, args.load_model, model_dir)

    #     # Create evaluators and early_stoppings
    #     # evaluators: for different eval_set
    #     self.evaluators = {}

    #     if args.eval_epoch is not None:
    #         self.evaluators = create_evaluators(
    #             self.dataset, args.eval_types, args.metrics, args.ks, args.n_thread
    #         )
    #         self.early_stopping = EarlyStopping(args.early_stop)

    #     # Start training and evaluation.
    #     print_seperate_line()
    #     if args.eval_epoch is not None:
    #         self.eval(args)
    #         print_seperate_line()

    #     self.int_weight_v = args.int_weight
    #     self.pop_weight_v = args.pop_weight

    #     for epoch in range(1, args.epoch + 1):
    #         self.train_1_epoch(epoch, args)
    #         self.int_weight_v *= args.loss_decay
    #         self.pop_weight_v *= args.loss_decay
    #         self.sampler.margin *= args.margin_decay

    #         if args.eval_epoch is not None and epoch % args.eval_epoch == 0:
    #             print_seperate_line()
    #             self.eval(args)
    #             print_seperate_line()

    #             if self.early_stopping.check_stop(self.evaluators, epoch):
    #                 break

    #     print(self.early_stopping)
    #     print_seperate_line()

    #     # Save model.
    #     if args.save_model:
    #         save_model(self.sess, self.saver, args.verbose_name, args.epoch, model_dir)

    def create_variables(self):
        self.all_embeds_0 = tf.compat.v1.get_variable(
            "all_embeds_0",
            shape=[self.dataset.n_user + self.dataset.n_item, self.embed_size],
            initializer=self.initializer,
        )
        self.u_embeds_0, self.i_embeds_0 = tf.split(
            self.all_embeds_0, [self.dataset.n_user, self.dataset.n_item], 0
        )
        self.int_embeds_0, self.pop_embeds_0 = tf.split(self.all_embeds_0, 2, 1)

        self.int_item_bias = tf.compat.v1.get_variable(
            "int_item_bias",
            shape=[self.n_item, 1],
            initializer=tf.compat.v1.zeros_initializer,
        )
        self.pop_item_bias = tf.compat.v1.get_variable(
            "pop_item_bias",
            shape=[self.n_item, 1],
            initializer=tf.compat.v1.zeros_initializer,
        )

        embed_size = self.embed_size // 2
        # if args.embed_type == "ngcf":
        #     self.int_W1s = []
        #     self.int_b1s = []
        #     self.int_W2s = []
        #     self.int_b2s = []
        #     self.pop_W1s = []
        #     self.pop_b1s = []
        #     self.pop_W2s = []
        #     self.pop_b2s = []
        #     for i in range(args.n_layer):
        #         self.int_W1s.append(
        #             tf.compat.v1.get_variable(
        #                 "int_W1_{}".format(i),
        #                 shape=[embed_size, embed_size],
        #                 initializer=self.initializer,
        #             )
        #         )
        #         self.int_b1s.append(
        #             tf.compat.v1.get_variable(
        #                 "int_b1_{}".format(i),
        #                 shape=[1, embed_size],
        #                 initializer=self.initializer,
        #             )
        #         )
        #         self.int_W2s.append(
        #             tf.compat.v1.get_variable(
        #                 "int_W2_{}".format(i),
        #                 shape=[embed_size, embed_size],
        #                 initializer=self.initializer,
        #             )
        #         )
        #         self.int_b2s.append(
        #             tf.compat.v1.get_variable(
        #                 "int_b2_{}".format(i),
        #                 shape=[1, embed_size],
        #                 initializer=self.initializer,
        #             )
        #         )
        #         self.pop_W1s.append(
        #             tf.compat.v1.get_variable(
        #                 "pop_W1_{}".format(i),
        #                 shape=[embed_size, embed_size],
        #                 initializer=self.initializer,
        #             )
        #         )
        #         self.pop_b1s.append(
        #             tf.compat.v1.get_variable(
        #                 "pop_b1_{}".format(i),
        #                 shape=[1, embed_size],
        #                 initializer=self.initializer,
        #             )
        #         )
        #         self.pop_W2s.append(
        #             tf.compat.v1.get_variable(
        #                 "pop_W2_{}".format(i),
        #                 shape=[embed_size, embed_size],
        #                 initializer=self.initializer,
        #             )
        #         )
        #         self.pop_b2s.append(
        #             tf.compat.v1.get_variable(
        #                 "pop_b2_{}".format(i),
        #                 shape=[1, embed_size],
        #                 initializer=self.initializer,
        #             )
        #         )

        # if args.inference_type == "mlp":
        #     weight_sizes = [2 * embed_size] + [x // 2 for x in args.weight_sizes]
        #     self.int_Ws = []
        #     self.int_bs = []
        #     self.pop_Ws = []
        #     self.pop_bs = []
        #     for i in range(len(args.weight_sizes)):
        #         self.int_Ws.append(
        #             tf.compat.v1.get_variable(
        #                 "int_W_{}".format(i),
        #                 shape=[weight_sizes[i], weight_sizes[i + 1]],
        #                 initializer=self.initializer,
        #             )
        #         )
        #         self.int_bs.append(
        #             tf.compat.v1.get_variable(
        #                 "int_b_{}".format(i),
        #                 shape=[1, weight_sizes[i + 1]],
        #                 initializer=self.initializer,
        #             )
        #         )
        #     self.int_h = tf.compat.v1.get_variable(
        #         "int_h",
        #         shape=[weight_sizes[-1] + embed_size, 1],
        #         initializer=self.initializer,
        #     )
        #     for i in range(len(args.weight_sizes)):
        #         self.pop_Ws.append(
        #             tf.compat.v1.get_variable(
        #                 "pop_W_{}".format(i),
        #                 shape=[weight_sizes[i], weight_sizes[i + 1]],
        #                 initializer=self.initializer,
        #             )
        #         )
        #         self.pop_bs.append(
        #             tf.compat.v1.get_variable(
        #                 "pop_b_{}".format(i),
        #                 shape=[1, weight_sizes[i + 1]],
        #                 initializer=self.initializer,
        #             )
        #         )
        #     self.pop_h = tf.compat.v1.get_variable(
        #         "pop_h",
        #         shape=[weight_sizes[-1] + embed_size, 1],
        #         initializer=self.initializer,
        #     )

    def create_embeds(self):
        s_norm_adj = create_norm_adj(
            self.dataset.u_interacts,
            self.dataset.i_interacts,
            self.dataset.n_user,
            self.dataset.n_item,
        )
        # if args.embed_type == "ngcf":
        #     self.int_embeds = create_ngcf_embed(
        #         self.int_embeds_0,
        #         s_norm_adj,
        #         args.n_layer,
        #         self.int_W1s,
        #         self.int_b1s,
        #         self.int_W2s,
        #         self.int_b2s,
        #         args,
        #     )
        #     self.pop_embeds = create_ngcf_embed(
        #         self.pop_embeds_0,
        #         s_norm_adj,
        #         args.n_layer,
        #         self.pop_W1s,
        #         self.pop_b1s,
        #         self.pop_W2s,
        #         self.pop_b2s,
        #         args,
        #     )
        # elif args.embed_type == "lightgcn":
        self.int_embeds = create_lightgcn_embed(
            self.int_embeds_0, s_norm_adj, 0 # args.n_layer
        )
        self.pop_embeds = create_lightgcn_embed(
            self.pop_embeds_0, s_norm_adj, 0 # args.n_layer
        )
        self.all_embeds = tf.concat([self.int_embeds, self.pop_embeds], 1)
        self.u_embeds, self.i_embeds = tf.split(
            self.all_embeds, [self.dataset.n_user, self.dataset.n_item], 0
        )

    def create_loss(self):
        self.batch_pos_i = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        self.batch_neg_i = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        self.batch_neg_mask = tf.compat.v1.placeholder(tf.float32, shape=(None,))
        batch_pos_i_embeds = tf.nn.embedding_lookup(self.i_embeds, self.batch_pos_i)
        batch_neg_i_embeds = tf.nn.embedding_lookup(self.i_embeds, self.batch_neg_i)
        users_int, users_pop = tf.split(self.batch_u_embeds, 2, 1)
        items_p_int, items_p_pop = tf.split(batch_pos_i_embeds, 2, 1)
        items_n_int, items_n_pop = tf.split(batch_neg_i_embeds, 2, 1)

        p_bias_items_int = tf.nn.embedding_lookup(self.int_item_bias, self.batch_pos_i)
        n_bias_items_int = tf.nn.embedding_lookup(self.int_item_bias, self.batch_neg_i)
        p_bias_items_pop = tf.nn.embedding_lookup(self.pop_item_bias, self.batch_pos_i)
        n_bias_items_pop = tf.nn.embedding_lookup(self.pop_item_bias, self.batch_neg_i)
        # if args.inference_type == "inner_product":
        p_score_int = inner_product(users_int, items_p_int) + tf.squeeze(p_bias_items_int)
        n_score_int = inner_product(users_int, items_n_int) + tf.squeeze(n_bias_items_int)
        p_score_pop = inner_product(users_pop, items_p_pop) + tf.squeeze(p_bias_items_pop)
        n_score_pop = inner_product(users_pop, items_n_pop) + tf.squeeze(n_bias_items_pop)

        # elif args.inference_type == "mlp":
        #     p_score_int = mlp(
        #         users_int, items_p_int, self.int_Ws, self.int_bs, self.int_h, args
        #     )
        #     n_score_int = mlp(
        #         users_int, items_n_int, self.int_Ws, self.int_bs, self.int_h, args
        #     )
        #     p_score_pop = mlp(
        #         users_pop, items_p_pop, self.pop_Ws, self.pop_bs, self.pop_h, args
        #     )
        #     n_score_pop = mlp(
        #         users_pop, items_n_pop, self.pop_Ws, self.pop_bs, self.pop_h, args
        #     )

        p_score_total = p_score_int + p_score_pop
        n_score_total = n_score_int + n_score_pop

        self.loss_int = mask_bpr_loss(p_score_int, n_score_int, self.batch_neg_mask)
        # The following should correspondon to conformity loss (O1+O2) from https://arxiv.org/pdf/2006.11011.pdf
        # But it does not: Switching around n_score_pop and p_score_pop does not equate the negative sign in the paper.
        # However, this is consistent with the original implementation
        self.loss_pop = mask_bpr_loss(
            n_score_pop, p_score_pop, self.batch_neg_mask
        ) + mask_bpr_loss(p_score_pop, n_score_pop, 1 - self.batch_neg_mask)
        self.loss_total = bpr_loss(p_score_total, n_score_total)

        user_int = tf.concat([users_int, users_int], 0)
        user_pop = tf.concat([users_pop, users_pop], 0)
        item_int = tf.concat([items_p_int, items_n_int], 0)
        item_pop = tf.concat([items_p_pop, items_n_pop], 0)
        self.discrepency_loss = discrepency_loss(item_int, item_pop) + discrepency_loss(
            user_int, user_pop
        )
        self.mf_loss = (
            self.int_weight * self.loss_int
            + self.pop_weight * self.loss_pop
            + self.loss_total
            - self.dis_pen * self.discrepency_loss
        )

        self.reg_loss = self.reg * l2_embed_loss(self.all_embeds)
        # if args.embed_type == "ngcf":
        #     for x in (
        #         self.int_W1s
        #         + self.int_b1s
        #         + self.int_W2s
        #         + self.int_b2s
        #         + self.pop_W1s
        #         + self.pop_b1s
        #         + self.pop_W2s
        #         + self.pop_b2s
        #     ):
        #         self.reg_loss += args.weight_reg * tf.nn.l2_loss(x)
        # if args.inference_type == "mlp":
        #     for x in (
        #         self.int_Ws
        #         + self.int_bs
        #         + [self.int_h]
        #         + self.pop_Ws
        #         + self.pop_bs
        #         + [self.pop_h]
        #     ):
        #         self.reg_loss += args.weight_reg * tf.nn.l2_loss(x)

        self.loss = self.mf_loss + self.reg_loss

    def create_batch_ratings(self):
        self.batch_u = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        self.batch_u_embeds = tf.nn.embedding_lookup(self.u_embeds, self.batch_u)
        users_int, _ = tf.split(self.batch_u_embeds, 2, 1)
        i_int_embeds, _ = tf.split(self.i_embeds, 2, 1)

        # if args.inference_type == "inner_product":
        self.batch_ratings = tf.matmul(users_int, i_int_embeds, transpose_b=True) + tf.transpose(self.int_item_bias)
        self.users_embs_all = users_int
        self.items_embs_all = i_int_embeds
        # elif args.inference_type == "mlp":
        #     u_size = tf.shape(users_int)[0]
        #     i_size = tf.shape(i_int_embeds)[0]
        #     u_repeats = tf.repeat(users_int, i_size, axis=0)
        #     i_tiles = tf.tile(i_int_embeds, [u_size, 1])
        #     scores = mlp(u_repeats, i_tiles, self.int_Ws, self.int_bs, self.int_h, args)
        #     self.batch_ratings = tf.reshape(scores, [u_size, i_size])

    # def train_1_epoch(self, epoch, args):
    #     self.timer.start("Epoch {}".format(epoch))

    #     losses = []
    #     mf_losses = []
    #     dis_losses = []
    #     reg_losses = []
    #     for users, pos_items, neg_items, neg_mask in self.sampler.sample():
    #         (
    #             _,
    #             batch_loss,
    #             batch_mf_loss,
    #             batch_dis_loss,
    #             batch_reg_loss,
    #         ) = self.sess.run(
    #             [
    #                 self.opt,
    #                 self.loss,
    #                 self.mf_loss,
    #                 self.discrepency_loss,
    #                 self.reg_loss,
    #             ],
    #             feed_dict={
    #                 self.int_weight: self.int_weight_v,
    #                 self.pop_weight: self.pop_weight_v,
    #                 self.batch_u: users,
    #                 self.batch_pos_i: pos_items,
    #                 self.batch_neg_i: neg_items,
    #                 self.batch_neg_mask: neg_mask,
    #             },
    #         )
    #         losses.append(batch_loss)
    #         mf_losses.append(batch_mf_loss)
    #         dis_losses.append(batch_dis_loss)
    #         reg_losses.append(batch_reg_loss)

    #     print(
    #         "int_weight = {:.5f}, pop_weight = {:.5f}, margin = {:.5f}".format(
    #             self.int_weight_v, self.pop_weight_v, self.sampler.margin
    #         )
    #     )
    #     self.timer.stop(
    #         "loss = {:.5f} = {:.5f} (dis_loss = {:.5f}) + {:.5f}".format(
    #             np.mean(losses),
    #             np.mean(mf_losses),
    #             np.mean(dis_losses),
    #             np.mean(reg_losses),
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



# DICE Sampler from the original implementation at https://github.com/tsinghua-fib-lab/DICE/blob/390c73b23bca7021bec27f0c5ebf3a535a68ddb8/src/utils.py#L34
# We did not use DICE in this paper, because it tackles a bias that is more similar to the bandwagon effect, which we cannot have in our data. It does not tackle exposure bias.
import numpy as np
import scipy.sparse as sp
from Lee_2022_BISER.custom_utils import tocsr, csr_to_user_dict

class Sampler(object):

    def __init__(self, train, num_users, num_items, neg_sample_rate, val=None):

        # self.name = flags_obj.name + '_sampler'
        self.train = train
        # self.lil_record = lil_record
        # self.record = list(dok_record.keys())

        tr_pos_index = np.where(train[:,2] > 0.5)[0]
        train_pos = train[tr_pos_index]
        self.popularity = np.unique(train_pos[:,1], return_counts=True)[1] # array of numbers of positive interactions per item
        # self.record contains pairs of [user, item]
        self.record = train_pos[:,:2]
        # self.lil_record is equivalent to a dictionary that contains a list of user's positive items
        self.lil_record = csr_to_user_dict(tocsr(train_pos, num_users, num_items))

        if not val is None:
            # self.train_val = np.concatenate([train, val])
            # tr_val_pos_index = np.where(self.train_val[:,2] > 0.5)[0]
            # train_val_pos = self.train_val[tr_val_pos_index]
            # self.popularity_val = np.unique(train_val_pos[:,1], return_counts=True)[1] # array of numbers of positive interactions per item

            self.val = val
            val_pos_index = np.where(self.val[:,2] > 0.5)[0]
            val_pos = self.val[val_pos_index]
            self.record_val = val_pos[:,:2]
            self.lil_record_val = csr_to_user_dict(tocsr(val_pos, num_users, num_items))


        self.neg_sample_rate = neg_sample_rate
        self.n_user = num_users # lil_record.shape[0]
        self.n_item = num_items # lil_record.shape[1]

        # For the CPR sampler
        self.users_train = [user for user in self.lil_record.keys() if np.min(self.lil_record[user]) < 50]
        self.users_val = list(set(self.lil_record.keys()).difference(self.users_train))

    def sample(self, index, **kwargs):

        raise NotImplementedError
    
    def sample_val(self, index, **kwargs):

        raise NotImplementedError
    
    def get_pos_user_item(self, index):

        user = self.record[index][0]
        pos_item = self.record[index][1]

        return user, pos_item
    
    def get_pos_user_item_val(self, index):

        user = self.record_val[index][0]
        pos_item = self.record_val[index][1]

        return user, pos_item
    
    def generate_negative_samples(self, user, **kwargs):

        # negative_samples = np.full(self.neg_sample_rate, -1, dtype=np.int64)

        # user_pos = self.lil_record[user]
        # for count in range(self.neg_sample_rate):

        #     item = np.random.randint(self.n_item)
        #     while item in user_pos or item in negative_samples:
        #         item = np.random.randint(self.n_item)
        #     negative_samples[count] = item
        
        # return negative_samples

        raise NotImplementedError
    
    def generate_negative_samples_val(self, user, **kwargs):

        raise NotImplementedError


class DICESampler(Sampler):

    def __init__(self, train, num_users, num_items, neg_sample_rate, margin=10, pool=10, val=None):

        super(DICESampler, self).__init__(train, num_users, num_items, neg_sample_rate, val=val)
        # self.popularity = popularity 
        self.margin = margin
        self.pool = pool

        self.popularity_A_and_B = self.popularity
        self.popularity_A = self.popularity[50:]
        self.popularity_B = self.popularity[:50]
        del self.popularity

    def adapt(self, decay):

        self.margin = self.margin*decay

    def generate_negative_samples(self, user, pos_item):
        item_from_set_A = (pos_item >= 50)
        if item_from_set_A:
            self.popularity = self.popularity_A
        else:
            self.popularity = self.popularity_B

        negative_samples = np.full(self.neg_sample_rate, -1, dtype=np.int64)
        mask_type = np.full(self.neg_sample_rate, False, dtype=bool)

        user_pos = self.lil_record[user]

        item_pos_pop = self.popularity_A_and_B[pos_item]

        pop_items = np.nonzero(self.popularity > item_pos_pop + self.margin)[0]
        if item_from_set_A:
            pop_items += 50
        pop_items = pop_items[np.logical_not(np.isin(pop_items, user_pos))]
        num_pop_items = len(pop_items)

        # unpop_items = np.nonzero(self.popularity < item_pos_pop - 10)[0] # artefact from the original implementation
        unpop_items = np.nonzero(self.popularity < item_pos_pop/2)[0]
        if item_from_set_A:
            unpop_items += 50
        unpop_items = unpop_items[np.logical_not(np.isin(unpop_items, user_pos))]
        num_unpop_items = len(unpop_items)

        if num_pop_items < self.pool and num_unpop_items < self.neg_sample_rate: # necessary because the original implementation did not consider this case (none of the following cases tackles this)
            unpop_items = np.nonzero(self.popularity < item_pos_pop)[0] # relax margin
            if item_from_set_A:
                unpop_items += 50
            unpop_items = unpop_items[np.logical_not(np.isin(unpop_items, user_pos))]
            num_unpop_items = len(unpop_items)

        if num_unpop_items < self.pool and num_pop_items < self.neg_sample_rate: # necessary because the original implementation did not consider this case (none of the following cases tackles this)
            pop_items = np.nonzero(self.popularity > item_pos_pop)[0] # relax margin
            if item_from_set_A:
                pop_items += 50
            pop_items = pop_items[np.logical_not(np.isin(pop_items, user_pos))]
            num_pop_items = len(pop_items)

        if num_pop_items < self.pool:
            if num_unpop_items < self.neg_sample_rate:
                raise(Exception)
            
            for count in range(self.neg_sample_rate):

                # index = np.random.randint(num_unpop_items)
                # item = unpop_items[index]
                # while item in negative_samples:
                #     index = np.random.randint(num_unpop_items)
                #     item = unpop_items[index]

                remaining = list(set(unpop_items).difference(negative_samples))
                index = np.random.randint(len(remaining))
                item = remaining[index]

                negative_samples[count] = item
                mask_type[count] = False

        elif num_unpop_items < self.pool:
            if num_pop_items < self.neg_sample_rate:
                raise(Exception)
            
            for count in range(self.neg_sample_rate):

                # index = np.random.randint(num_pop_items)
                # item = pop_items[index]
                # while item in negative_samples:
                #     index = np.random.randint(num_pop_items)
                #     item = pop_items[index]

                remaining = list(set(pop_items).difference(negative_samples))
                index = np.random.randint(len(remaining))
                item = remaining[index]

                negative_samples[count] = item
                mask_type[count] = True
        
        else:
            if num_pop_items < self.neg_sample_rate or num_unpop_items < self.neg_sample_rate: # necessary because the original implementation did not consider this case (none of the following cases tackles this)
                raise ValueError("Pool must be larger or equal than negative_sample_size.")

            for count in range(self.neg_sample_rate):

                if np.random.random() < 0.5:

                    # index = np.random.randint(num_pop_items)
                    # item = pop_items[index]
                    # while item in negative_samples:
                    #     index = np.random.randint(num_pop_items)
                    #     item = pop_items[index]

                    remaining = list(set(pop_items).difference(negative_samples))
                    index = np.random.randint(len(remaining))
                    item = remaining[index]

                    negative_samples[count] = item
                    mask_type[count] = True

                else:

                    # index = np.random.randint(num_unpop_items)
                    # item = unpop_items[index]
                    # while item in negative_samples:
                    #     index = np.random.randint(num_unpop_items)
                    #     item = unpop_items[index]

                    remaining = list(set(unpop_items).difference(negative_samples))
                    index = np.random.randint(len(remaining))
                    item = remaining[index]

                    negative_samples[count] = item
                    mask_type[count] = False

        return negative_samples, mask_type

    def generate_negative_samples_val(self, user, pos_item):

        negative_samples = np.full(self.neg_sample_rate, -1, dtype=np.int64)
        mask_type = np.full(self.neg_sample_rate, False, dtype=bool)

        user_pos = self.lil_record_val[user]

        item_pos_pop = self.popularity_val[pos_item]
        popularity_val_B = self.popularity_val[:50]

        pop_items = np.nonzero(popularity_val_B > item_pos_pop + self.margin)[0]
        pop_items = pop_items[np.logical_not(np.isin(pop_items, user_pos))]
        num_pop_items = len(pop_items)

        # unpop_items = np.nonzero(self.popularity_val < item_pos_pop - 10)[0] # artefact from the original implementation
        unpop_items = np.nonzero(popularity_val_B < item_pos_pop/2)[0]
        unpop_items = unpop_items[np.logical_not(np.isin(unpop_items, user_pos))]
        num_unpop_items = len(unpop_items)

        if num_pop_items < self.pool and num_unpop_items < self.neg_sample_rate: # necessary because the original implementation did not consider this case (none of the following cases tackles this)
            unpop_items = np.nonzero(popularity_val_B < item_pos_pop)[0]
            unpop_items = unpop_items[np.logical_not(np.isin(unpop_items, user_pos))]
            num_unpop_items = len(unpop_items)

        if num_unpop_items < self.pool and num_pop_items < self.neg_sample_rate: # necessary because the original implementation did not consider this case (none of the following cases tackles this)
            pop_items = np.nonzero(popularity_val_B > item_pos_pop)[0]
            pop_items = pop_items[np.logical_not(np.isin(pop_items, user_pos))]
            num_pop_items = len(pop_items)

        if num_pop_items < self.pool:
            if num_unpop_items < self.neg_sample_rate:
                raise(Exception)
            
            for count in range(self.neg_sample_rate):

                # index = np.random.randint(num_unpop_items)
                # item = unpop_items[index]
                # while item in negative_samples:
                #     index = np.random.randint(num_unpop_items)
                #     item = unpop_items[index]

                remaining = list(set(unpop_items).difference(negative_samples))
                index = np.random.randint(len(remaining))
                item = remaining[index]

                negative_samples[count] = item
                mask_type[count] = False

        elif num_unpop_items < self.pool:
            if num_pop_items < self.neg_sample_rate:
                raise(Exception)
            
            for count in range(self.neg_sample_rate):

                # index = np.random.randint(num_pop_items)
                # item = pop_items[index]
                # while item in negative_samples:
                #     index = np.random.randint(num_pop_items)
                #     item = pop_items[index]

                remaining = list(set(pop_items).difference(negative_samples))
                index = np.random.randint(len(remaining))
                item = remaining[index]

                negative_samples[count] = item
                mask_type[count] = True
        
        else:
            if num_pop_items < self.neg_sample_rate or num_unpop_items < self.neg_sample_rate:
                raise(Exception)

            for count in range(self.neg_sample_rate):

                if np.random.random() < 0.5:

                    # index = np.random.randint(num_pop_items)
                    # item = pop_items[index]
                    # while item in negative_samples:
                    #     index = np.random.randint(num_pop_items)
                    #     item = pop_items[index]

                    remaining = list(set(pop_items).difference(negative_samples))
                    index = np.random.randint(len(remaining))
                    item = remaining[index]

                    negative_samples[count] = item
                    mask_type[count] = True

                else:

                    # index = np.random.randint(num_unpop_items)
                    # item = unpop_items[index]
                    # while item in negative_samples:
                    #     index = np.random.randint(num_unpop_items)
                    #     item = unpop_items[index]

                    remaining = list(set(unpop_items).difference(negative_samples))
                    index = np.random.randint(len(remaining))
                    item = remaining[index]

                    negative_samples[count] = item
                    mask_type[count] = False

        return negative_samples, mask_type

    def sample(self, index):

        user, pos_item = self.get_pos_user_item(index)

        users = np.full(self.neg_sample_rate, user, dtype=np.int64)
        items_pos = np.full(self.neg_sample_rate, pos_item, dtype=np.int64)
        items_neg, mask_type = self.generate_negative_samples(user, pos_item=pos_item)

        return users, items_pos, items_neg, mask_type

    def sample_val(self, index):

        user, pos_item = self.get_pos_user_item_val(index)

        users = np.full(self.neg_sample_rate, user, dtype=np.int64)
        items_pos = np.full(self.neg_sample_rate, pos_item, dtype=np.int64)
        items_neg, mask_type = self.generate_negative_samples_val(user, pos_item=pos_item)

        return users, items_pos, items_neg, mask_type