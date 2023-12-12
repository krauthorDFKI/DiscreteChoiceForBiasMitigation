from typing import Tuple

import numpy as np
import tensorflow as tf

from Lee_2022_BISER.models.uae import uAE
from Lee_2022_BISER.models.iae import iAE
from Lee_2022_BISER.models.proposed import Proposed


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ae_trainer(sess: tf.compat.v1.Session, data, train: np.ndarray, val: np.ndarray, test: np.ndarray, num_users: int, num_items: int,
                   n_components: int, eta, lam: float, max_iters, batch_size,
                   model_name: str, item_freq: np.ndarray,
                   unbiased_eval: bool, random_state: int, wu=0.1, wi=0.1, n_early_stop: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """Train autoencoder models."""
    # if model_name == 'uae':
    #     model = uAE(sess, data,  train, val, test, num_users, num_items, hidden_dim=n_components, eta=eta, random_state=random_state,
    #                 reg=lam, max_iters=max_iters, batch_size=batch_size)
    #     model.train_model(pscore=item_freq, unbiased_eval=unbiased_eval)

    # elif model_name == 'iae':
    #     model = iAE(sess, data, train, val, test, num_users, num_items, hidden_dim=n_components, eta=eta, random_state=random_state,
    #                 reg=lam, max_iters=max_iters, batch_size=batch_size)
    #     model.train_model(pscore=item_freq, unbiased_eval=unbiased_eval)

    # elif model_name == 'proposed':
    model = Proposed(sess, data, train, val, test, num_users, num_items, hidden_dim=n_components, eta=eta, random_state=random_state,
                reg=lam, max_iters=max_iters, batch_size=batch_size, wu=wu, wi=wi)
    train_losses, val_losses = model.train_model(pscore=item_freq, unbiased_eval=unbiased_eval, n_early_stop = n_early_stop)
    return model.best_weights_enc_u, model.best_weights_dec_u, model.best_bias_enc_u, model.best_bias_dec_u, model.best_weights_enc_i, model.best_weights_dec_i, model.best_bias_enc_i, model.best_bias_dec_i, train_losses, val_losses

    # return model.best_weights_enc, model.best_weights_dec, model.best_bias_enc, model.best_bias_dec
