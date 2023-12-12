import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops
from tqdm import tqdm

from Lee_2022_BISER.evaluate.evaluator import aoa_evaluator, unbiased_evaluator
from Lee_2022_BISER.models.recommenders import PointwiseRecommender, MACR

from Wan_2022_CPR.bpr import BPR
from Wan_2022_CPR.ubpr import UBPR
from Wan_2022_CPR.cpr import CPR
from Zhang_2021_PD.pd import PD

from Lee_2022_BISER.AE_trainer import ae_trainer
from Lee_2022_BISER.PointWise_trainer import pointwise_trainer, macr_trainer, cjmf_trainer


from Lee_2022_BISER.CPR_PD_PointWise_trainer import cpr_pointwise_trainer

class Trainer:
    """Trainer Class for ImplicitRecommender."""
    def __init__(self, hyperparams) -> None:
    # (self, data: str, random_state: list, hidden: int, date_now: str, max_iters: int = 1000, lam: float=1e-4, batch_size: int = 12, wu=0.1, wi=0.1, alpha: float = 0.5, \
    #             clip: float=0.1, eta: float=0.1, model_name: str='mf', unbiased_eval:bool = True, neg_sample: int=10, C: int=8, alpha_cjmf: float=220000, beta_cjmf: float=0.5, \
    #             macr_c: float=0.1, macr_alpha: float=0.1, macr_beta: float=0.1, best_model_save: bool=True) -> None:
        """Initialize class."""

        self.data = hyperparams.get("data", None)
        self.at_k = [1, 3, 5]
        self.dim = hyperparams.get("hidden", None)
        self.lam = hyperparams.get("lam", None)
        self.clip = hyperparams.get("clip", None) if hyperparams.get("model_name", None) == 'relmf' else 0
        self.batch_size = hyperparams.get("batch_size", None)
        self.max_iters = hyperparams.get("max_iters", None)
        self.eta = hyperparams.get("eta", None)
        self.model_name = hyperparams.get("model_name", None)
        self.unbiased_eval = hyperparams.get("unbiased_eval", None)
        self.n_early_stop = hyperparams.get("n_early_stop", None)

        self.wu = hyperparams.get("wu", None)
        self.wi = hyperparams.get("wi", None)
        self.pd_alpha = hyperparams.get("alpha", None)
        self.neg_sample = hyperparams.get("neg_sample", None)

        # cjmf
        self.C = hyperparams.get("C", None)
        self.alpha_cjmf = hyperparams.get("alpha_cjmf", None)
        self.beta_cjmf = hyperparams.get("beta_cjmf", None)

        # macr
        self.macr_c = hyperparams.get("macr_c", None)
        self.macr_alpha = hyperparams.get("macr_alpha", None)
        self.macr_beta = hyperparams.get("macr_beta", None)

        # pd
        self.pd_gamma = hyperparams.get("gamma", None)

        # (u)bpr
        self.ubpr_ps_pow = hyperparams.get("ps_pow", None)
        self.ubpr_clip_min = hyperparams.get("clip_min", None)

        # cpr
        self.cpr_beta = hyperparams.get("beta", None)
        self.cpr_max_k_interact = hyperparams.get("max_k_interact", None)
        self.cpr_gamma = hyperparams.get("gamma", None)
        self.n_step = hyperparams.get("n_step", None)

        # dice
        self.dice_init_margin = hyperparams.get("init_margin", None)
        self.dice_dis_pen = hyperparams.get("dis_pen", None)
        self.dice_init_int_weight = hyperparams.get("init_int_weight", None)
        self.dice_init_pop_weight = hyperparams.get("init_pop_weight", None)
        self.loss_decay = hyperparams.get("loss_decay", None)
        self.dice_margin_decay = hyperparams.get("margin_decay", None)
        self.dice_pool = hyperparams.get("pool", None)

        # self.best_model_save = hyperparams.get("best_model_save", None)
        # self.date_now = hyperparams.get("date_now", None)

        self.random_state = [r for r in range(1, int(hyperparams.get("random_state", None)) + 1)]

        # print("======================================================")
        # print("random state: ", self.random_state)
        # print("======================================================")

    def run(self, train, val, pscore, item_freq) -> None:
        # print("======================================================")
        # print("date: ", self.date_now)
        # print("======================================================")

        """Train pointwise implicit recommenders."""
        # self.train = np.load(f'./data/user_study/preprocessed/point_{self.alpha}/train.npy')
        # import os
        # if os.path.exists(f'./data/user_study/preprocessed/point_{self.alpha}/val.npy'):
        #     val = np.load(f'./data/user_study/preprocessed/point_{self.alpha}/val.npy')
        # else:
        #     val = None
        # pscore = np.load(f'./data/user_study/preprocessed/point_{self.alpha}/pscore.npy')
        # item_freq = np.load(f'./data/user_study/preprocessed/point_{self.alpha}/item_freq.npy')

        self.train = train

        self.num_users = np.int32(self.train[:, 0].max() + 1)
        self.num_items = np.int32(self.train[:, 1].max() + 1)
        """
        Add methods in AE models 
        """
        sub_results_sum = pd.DataFrame()
        for random_state in self.random_state: #tqdm(self.random_state, ascii=True): # The tqdm somehow causes stderr to contain some weird error. The results are unaffected
            # print("random seed now :", random_state)
            tf.compat.v1.reset_default_graph()
            ops.reset_default_graph()
            tf.compat.v1.set_random_seed(random_state)
            config = tf.compat.v1.ConfigProto(device_count={'GPU': 0}) # disable GPU (not necessary here)
            config.gpu_options.allow_growth = True
            sess = tf.compat.v1.Session(config=config)

            if self.model_name in ['proposed']: # BISER
                weights_enc_u, weights_dec_u, bias_enc_u, bias_dec_u, weights_enc_i, weights_dec_i, bias_enc_i, bias_dec_i, train_losses, val_losses = ae_trainer(sess, 
                            data=self.data, train=self.train, val=val, test=None, num_users=self.num_users, num_items=self.num_items, n_components=self.dim, wu=self.wu, 
                            wi=self.wi, eta=self.eta, lam=self.lam, max_iters=self.max_iters, batch_size=self.batch_size, model_name=self.model_name, item_freq=item_freq,
                            unbiased_eval = self.unbiased_eval, random_state=random_state, n_early_stop=self.n_early_stop)
                self.weights_enc_u = weights_enc_u
                self.weights_dec_u = weights_dec_u
                self.bias_enc_u = bias_enc_u
                self.bias_dec_u = bias_dec_u
                self.weights_enc_i = weights_enc_i
                self.weights_dec_i = weights_dec_i
                self.bias_enc_i = bias_enc_i
                self.bias_dec_i = bias_dec_i

            elif self.model_name in ['mf', 'relmf']:
                model = PointwiseRecommender(model_name=self.model_name,
                    num_users=self.num_users, num_items=self.num_items,
                    clip=self.clip, dim=self.dim, lam=self.lam, eta=self.eta)

                u_emb, i_emb, train_losses, val_losses = pointwise_trainer(
                    sess, data=self.data, model=model, train=self.train, val=val, test=None, 
                    num_users=self.num_users, num_items=self.num_items, pscore=pscore,
                    max_iters=self.max_iters, batch_size=self.batch_size, item_freq=item_freq, unbiased_eval = self.unbiased_eval,
                    model_name=self.model_name, date_now=None, n_early_stop=self.n_early_stop)
                self.u_emb = u_emb
                self.i_emb = i_emb

            # elif self.model_name in ['cjmf']:
            #     u_emb, i_emb, train_losses, val_losses = cjmf_trainer(sess=sess, data=self.data, n_components=self.dim, num_users=self.num_users, num_items=self.num_items, \
            #                                 batch_size=self.batch_size, max_iters=self.max_iters, item_freq=item_freq, \
            #                                 unbiased_eval=self.unbiased_eval, C=self.C,  lr=self.eta, reg=self.lam, \
            #                                 alpha=self.alpha_cjmf, beta=self.beta_cjmf, train=self.train, val=val, seed=random_state, model_name=self.model_name)
            #     self.u_emb = u_emb
            #     self.i_emb = i_emb

            elif self.model_name in ['macr']:
                model = MACR(model_name=self.model_name, 
                    num_users=self.num_users, num_items=self.num_items,
                    dim=self.dim, lam=self.lam, eta=self.eta, batch_size=self.batch_size, c=self.macr_c, alpha=self.macr_alpha, beta=self.macr_beta)

                u_emb, i_emb, train_losses, val_losses = macr_trainer(
                    sess, data=self.data, model=model, train=self.train, val = val, test=None, 
                    num_users=self.num_users, num_items=self.num_items, pscore=pscore,
                    max_iters=self.max_iters, batch_size=self.batch_size, item_freq=item_freq, unbiased_eval = self.unbiased_eval,
                    model_name=self.model_name, date_now=None, neg_sample=self.neg_sample, n_early_stop=self.n_early_stop)
                self.u_emb = u_emb
                self.i_emb = i_emb

            elif self.model_name in ['bpr', 'ubpr', 'dice', 'pd', 'cpr']:
                if self.model_name == 'bpr':
                    model_class = BPR(train=self.train, num_users=self.num_users, num_items=self.num_items, dim=self.dim, lam=self.lam, eta=self.eta)
                elif self.model_name == 'ubpr':
                    model_class = UBPR(train=self.train, num_users=self.num_users, num_items=self.num_items, dim=self.dim, lam=self.lam, eta=self.eta, 
                        ps_pow=self.ubpr_ps_pow, clip_min=self.ubpr_clip_min)
                elif self.model_name == 'pd':
                    model_class = PD(num_users=self.num_users, num_items=self.num_items, batch_size=self.batch_size, dim=self.dim, lam=self.lam, eta=self.eta, gamma=self.pd_gamma, 
                    alpha=self.pd_alpha)
                elif self.model_name == 'cpr':
                    model_class = CPR(train=self.train, val=val, num_users=self.num_users, num_items=self.num_items, batch_size=self.batch_size, dim=self.dim, lam=self.lam, eta=self.eta,
                                batch_total_sample_sizes = self.batch_size, beta=self.cpr_beta, max_k_interact=self.cpr_max_k_interact, gamma = self.cpr_gamma, 
                                n_step = self.n_step, seed=0)
                
                u_emb, i_emb, train_losses, val_losses = cpr_pointwise_trainer(
                    sess, data=self.data, model=model_class, train=self.train, val=val, test=None, 
                    num_users=self.num_users, num_items=self.num_items, pscore=pscore,
                    max_iters=self.max_iters, batch_size=self.batch_size, item_freq=item_freq, unbiased_eval = self.unbiased_eval,
                    model_name=self.model_name, date_now=None, neg_sample=self.neg_sample, n_early_stop=self.n_early_stop)
                self.u_emb = u_emb
                self.i_emb = i_emb


        if self.model_name in ['proposed']:
            self.evaluator = aoa_evaluator(user_embed=[self.weights_enc_u, self.weights_dec_u, self.weights_enc_i, self.weights_dec_i], 
                                        item_embed=[self.bias_enc_u, self.bias_dec_u, self.bias_enc_i, self.bias_dec_i],
                                        num_users=self.num_users, num_items=self.num_items, model_name=self.model_name, train=self.train)
        else:
            self.evaluator = aoa_evaluator(user_embed=self.u_emb, item_embed=self.i_emb,
                                        num_users=self.num_users, num_items=self.num_items, model_name=self.model_name, train=self.train)

        return train_losses, val_losses

    def predict(self, user, items):
        return self.evaluator.predict(user, items)