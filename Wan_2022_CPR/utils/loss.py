import tensorflow as tf


def bpr_loss(pos_scores, neg_scores):
    pairwise_obj = pos_scores - neg_scores
    loss = tf.reduce_mean(tf.math.softplus(-pairwise_obj))
    return loss


def bce_loss(pos_scores, neg_scores):
    loss = - tf.reduce_mean(tf.concat([
        tf.math.log(tf.sigmoid(pos_scores))/3, 
        tf.math.log(1 - tf.sigmoid(neg_scores))], 
        axis=0))
    return loss


def ubpr_loss(u_embeds, i_embeds, i_bias, j_embeds, j_bias, i_p_scores, clip_min): # args):
    pairwise_obj = tf.reduce_sum(u_embeds * i_embeds - u_embeds * j_embeds, axis=1) + i_bias - j_bias
    # loss = (
    #     -(1 / i_p_scores) * (1 - (j_labels / j_p_scores)) * tf.compat.v1.log_sigmoid(pairwise_obj)
    # )
    # According to the UBPR Paper, j_labels should always be zero: https://dl.acm.org/doi/pdf/10.1145/3460231.3474274
    # In the original implementation, we can see that j_labels is always zero, so we do not need to sample it. We also do not need to calculate j_p_scores.
    # See in https://github.com/KhalilDMK/EBPR/blob/main/Code/engine_EBPR.py#L49:
    #     elif self.config['model'] == 'UBPR':
    #         loss = - ((pos_prediction - neg_prediction).sigmoid().log() / popularity_vector[pos_items]).sum()
    loss = - tf.compat.v1.log_sigmoid(pairwise_obj) / i_p_scores
    loss = tf.clip_by_value(loss, clip_value_min=clip_min, clip_value_max=10e5)
    loss = tf.reduce_mean(loss)
    return loss


def cpr_loss(pos_scores, neg_scores, sample_rate, batch_size): # , args):
    cpr_obj = pos_scores - neg_scores
    if sample_rate == 1:
        cpr_obj_neg = -cpr_obj
    else:
        cpr_obj_neg, _ = tf.math.top_k(-cpr_obj, k=batch_size, sorted=False)
    loss_train = tf.reduce_mean(tf.nn.softplus(cpr_obj_neg))
    loss_true = tf.reduce_mean(tf.nn.softplus(-cpr_obj))

    return loss_train, loss_true


def mask_bpr_loss(pos_scores, neg_scores, mask):
    loss = -tf.reduce_mean(mask * tf.math.log_sigmoid(pos_scores - neg_scores))
    return loss


def l2_embed_loss(*args):
    loss = 0
    for embeds in args:
        loss += tf.reduce_sum(tf.square(embeds), axis=1)
    return tf.reduce_mean(loss)


def discrepency_loss(x, y):
    # dcor has numerical problem when implemented by tensorflow
    # (loss would become nan),
    # so we use l2 loss instead.
    # return dcor(x, y)
    return tf.reduce_mean(tf.square(x - y))


def dcor(x, y):
    a = tf.norm(x[:, None] - x, axis=2)
    b = tf.norm(y[:, None] - y, axis=2)
    A = (
        a
        - tf.reduce_mean(a, axis=0)[None, :]
        - tf.reduce_mean(a, axis=1)[:, None]
        + tf.reduce_mean(a)
    )
    B = (
        b
        - tf.reduce_mean(b, axis=0)[None, :]
        - tf.reduce_mean(b, axis=1)[:, None]
        + tf.reduce_mean(b)
    )
    dcov2_xy = tf.reduce_mean(A * B)
    dcov2_xx = tf.reduce_mean(A * A)
    dcov2_yy = tf.reduce_mean(B * B)
    dcor = tf.sqrt(dcov2_xy) / tf.sqrt(tf.sqrt(dcov2_xx) * tf.sqrt(dcov2_yy))
    return dcor
