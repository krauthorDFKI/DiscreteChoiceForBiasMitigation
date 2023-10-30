import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from importlib import reload
import src.models

import warnings
import sys
sys.path.append("recommender/Models/Lee_2022_BISER")
sys.path.append("recommender/Models/Lee_2022_BISER/models/cpr_paper/")

from src.models.Lee_2022_BISER.trainer import Trainer

def prepare_data(users, alternatives, choices):
    return(np.asarray([[users[i], alternatives[i], choices[i]] for i in range(len(users))], dtype=object))

class Recommender:
    """Base class for all recommenders."""
    def __init__(self, hyperparams=None):
        pass
    
    def train(self, data_train, data_validation=None, hyperparams=None):
        """Trains the model on the given data."""
        result = {
            "train_losses": None,
            "validation_losses": None,
        }
        return result

    def predict(self, users, choiceset, rec_size=3):
        """Predicts the top-n recommendations for the given users on the given choiceset."""
        raise NotImplementedError()

    def clear(self):
        """Resets the environment. This is useful for models that use tensorflow, for example"""
        pass

class Recommender_random(Recommender):
    """Random Recommender"""
    def predict(self, users, choiceset, rec_size=3):
        return np.asarray([np.random.choice(choiceset, rec_size, replace=False) for u in users])

# k-nearest-neighbours
class Recommender_knn(Recommender):
    """k-nearest-neighbours Recommender"""
    def __init__(self, hyperparams=None):
        self.N = np.zeros((hyperparams["n_users"], hyperparams["n_alternatives"]))
        self.n_items = self.N.shape[1]
        self.similarities = None

        self.items_list_A = list(np.arange(50, 100))
        self.items_list_B = [i for i in range(hyperparams["n_alternatives"]) if i not in self.items_list_A]

        self.n_neighbours = hyperparams["n_neighbours"]
    
    def train(self, data_train, data_validation=None, hyperparams=None):
        self.N[data_train[:,0].astype(int), data_train[:,2].astype(int)] = 1
        
        # Calculate similarities only based on list A
        self.similarities = np.zeros((self.N.shape[0], self.N.shape[0]))
        N_list_A = self.N[:, self.items_list_A]

        lengths = np.linalg.norm(N_list_A, axis = 1, keepdims=True)
        self.similarities = np.matmul(N_list_A, N_list_A.transpose())
        self.similarities = self.similarities / lengths / lengths.transpose()

        result = {
            "train_losses": None,
            "validation_losses": None,
        }

        return result

    def predict(self, users, choiceset, rec_size=3):
        recommendations = []
        for user in users:
            allowed_neighbours = self.N.copy()

            if set(choiceset).intersection(self.items_list_A): # If only items from set A
                allowed_neighbours = np.unique(np.where(self.N[:, self.items_list_A] != 0)[0]) # Only users that have made observations on choice set
            else: # If also items from set B
                allowed_neighbours = np.unique(np.where(self.N[:, self.items_list_B] != 0)[0]) # Only users that have made observations on set B

            similarities = self.similarities[user, allowed_neighbours].copy()

            neighbours = allowed_neighbours[similarities.argsort()[::-1][: self.n_neighbours]]

            est_ratings = np.asarray([
                (sum(self.similarities[user, neighbours] * self.N[neighbours, item])
                 / sum(self.similarities[user, neighbours])) if item in choiceset else 0
                 for item in range(self.n_items)])
            
            import time
            s = time.time()

            # Generate list of top-n recommendations, break ties randomly
            top_n = []
            while len(top_n) < rec_size:
                not_picked_yet = [i for i in choiceset if i not in top_n]
                highest_rating = np.max(est_ratings[not_picked_yet])

                candidates = np.where(est_ratings == highest_rating)[0]
                candidates = [i for i in not_picked_yet if i in candidates] # Intersection with not_picked_yet
                top_n.append(np.random.choice(candidates))

            recommendations.append(top_n)
        
        return np.asarray(recommendations)


class Recommender_most_popular(Recommender):
    """Most Popular Recommender"""
    def __init__(self, hyperparams=None):
        self.ranking = None
        self.n_items = hyperparams["n_alternatives"]

    def train(self, data_train, data_validation=None, hyperparams=None):
        from collections import Counter
        unique_interactions = np.unique(data_train[:, [0, 2]].astype(int), axis=0)
        self.ranking = [item for item, c in Counter(unique_interactions[:,1]).most_common()]
        self.ranking = self.ranking + [i for i in range(self.n_items) if i not in self.ranking]

        result = {
            "train_losses": None,
            "validation_losses": None,
        }

        return result

    def predict(self, users, choiceset, rec_size=3):
        valids = [i for i in self.ranking if i in choiceset]
        preds = np.asarray([valids[:rec_size] for u in users])
        return(preds)

class Recommender_multivariate(Recommender):
    """Base class for all recommenders that use a multivariate model."""
    def __init__(self, hyperparams=None):
        self.model = None
        self.hyperparams = hyperparams

    def train(self, data_train, data_validation=None, hyperparams=None):
        data_train = self.model.generate_dataset(data_train[:,0],
                                            data_train[:,1],
                                            data_train[:,2], self.model.batch_size)
        if data_validation is not None:
            data_validation = self.model.generate_dataset(data_validation[:,0],
                                        data_validation[:,1],
                                        data_validation[:,2], self.model.batch_size)

        train_losses, validation_losses = self.model.train_steps(
            train_dataset=data_train,
            test_dataset=data_validation,
            learning_rate=self.hyperparams["learning_rate"],
            n_epochs=self.hyperparams["n_epochs"])

        del data_train
        if data_validation is not None:
            del data_validation

        result = {
            "train_losses": train_losses,
            "validation_losses": validation_losses,
        }
        return result

    def predict(self, users, choiceset, rec_size=None):
        preds = []
        for user in users:
            utilities = self.model(np.repeat(user, len(choiceset)), np.asarray(choiceset))
            utilities = np.asarray(utilities).reshape(-1)
            pred_user = [x for _, x in sorted(zip(utilities, choiceset), reverse=True)]
            preds.append(pred_user)

        preds = np.asarray(preds)
        # top rec_size
        if rec_size is not None:
            preds = preds[:, range(rec_size)]
        return(preds)

    def clear(self):
        import tensorflow as tf
        tf.keras.backend.clear_session()


class Recommender_multinomial_logit(Recommender_multivariate):
    """Multinomial Logit Model"""
    def __init__(self, hyperparams=None):
        from src.models.multinomial_logit import Recommender_Network

        # Run on CPU, because its faster for this small data set size
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')

        self.hyperparams = hyperparams
        self.model = Recommender_Network(n_users=hyperparams["n_users"], 
                                        n_items=hyperparams["n_alternatives"], 
                                        embedding_size=hyperparams["k"], 
                                        batch_size=hyperparams["batch_size"], 
                                        l2_embs = float(10**hyperparams["l2_embs_log10"]),
                                        n_early_stop = hyperparams["n_early_stop"],
                                    )

    def train(self, data_train, data_validation=None, hyperparams=None):
        reload(src.models.multinomial_logit)
        return super(Recommender_multinomial_logit, self).train(data_train, data_validation, hyperparams)


class Recommender_exponomial(Recommender_multivariate):
    """Exponomial Model"""
    def __init__(self, hyperparams=None):
        from src.models.exponomial import Recommender_Network

        # Run on CPU, because its faster for this small data set size
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')

        self.hyperparams = hyperparams
        self.model = Recommender_Network(n_users=hyperparams["n_users"], 
                                        n_items=hyperparams["n_alternatives"], 
                                        embedding_size=hyperparams["k"],
                                        batch_size=hyperparams["batch_size"], 
                                        l2_embs = float(10**hyperparams["l2_embs_log10"]),
                                        n_early_stop = hyperparams["n_early_stop"]
                                        )

    def train(self, data_train, data_validation=None, hyperparams=None):
        reload(src.models.exponomial)
        return super(Recommender_exponomial, self).train(data_train, data_validation, hyperparams)


class Recommender_generalized_multinomial_logit(Recommender_multivariate):
    """Generalized Multinomial Logit Model"""
    def __init__(self, hyperparams=None):
        from src.models.generalized_multinomial_logit import Recommender_Network
        self.hyperparams = hyperparams

        # Run on CPU, because its faster for this small data set size
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        
        self.model = Recommender_Network(
            n_users=hyperparams["n_users"], 
            n_items=hyperparams["n_alternatives"], 
            embedding_size=hyperparams["k"], 
            batch_size=hyperparams["batch_size"],
            n_classes=hyperparams["n_classes"], 
            l2_embs = float(10**hyperparams["l2_embs_log10"]),
            n_early_stop=hyperparams["n_early_stop"],
        )

    def train(self, data_train, data_validation=None, hyperparams=None):
        reload(src.models.generalized_multinomial_logit)
        return super(Recommender_generalized_multinomial_logit, self).train(data_train, data_validation, hyperparams)

class Recommender_binary_logit(Recommender_multivariate):
    """Binary Logit Model (Matrix Factorization)"""
    def __init__(self, hyperparams=None):
        from src.models.binary_logit import Recommender_Network
        self.hyperparams = hyperparams

        # Run on CPU, because its faster for this small data set size
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')

        self.model = Recommender_Network(n_users=hyperparams["n_users"], 
                                        n_items=hyperparams["n_alternatives"], 
                                        embedding_size=hyperparams["k"], 
                                        batch_size=hyperparams["batch_size"], 
                                        l2_embs = float(10**hyperparams["l2_embs_log10"]),
                                        n_early_stop=hyperparams["n_early_stop"],
        )

    def get_negativeInteractions(self, data_train, data_validation=None):
        # Positive interactions
        data_train_pos = np.asarray([data_train[:,0], data_train[:,2], np.ones(len(data_train[:,0]))]).transpose()
        if data_validation is not None:
            data_validation_pos = np.asarray([data_validation[:,0], data_validation[:,2], np.ones(len(data_validation[:,0]))]).transpose()
        else: 
            data_validation_pos = None

        data_train_neg = []
        for i in range(len(data_train)):
            user = data_train[i,0]
            choice = data_train[i,2]
            for option in data_train[i,1]:
                if option != choice:
                    data_train_neg.append([user, option, 0])
        data_train_neg = np.asarray(data_train_neg)

        data_train = np.concatenate([data_train_pos, data_train_neg])

        data_train = data_train.astype("int32")

        if data_validation is not None:
            data_validation_neg = []
            for i in range(len(data_validation)):
                user = data_validation[i,0]
                choice = data_validation[i,2]
                for option in data_validation[i,1]:
                    if option != choice:
                        data_validation_neg.append([user, option, 0])
            data_validation_neg = np.asarray(data_validation_neg)

            data_validation = np.concatenate([data_validation_pos, data_validation_neg])

            data_validation = data_validation.astype("int32")

        return data_train, data_validation

    def train(self, data_train, data_validation=None, hyperparams=None):
        reload(src.models.binary_logit)

        data_train_sampled, data_validation_sampled = self.get_negativeInteractions(data_train, data_validation)

        data_train = self.model.generate_dataset(data_train_sampled, self.model.batch_size)
        if data_validation is not None:
            data_validation = self.model.generate_dataset(data_validation_sampled, self.model.batch_size)

        train_losses, validation_losses = self.model.train_steps(
            train_dataset=data_train,
            test_dataset=data_validation,
            learning_rate=self.hyperparams["learning_rate"],
            n_epochs=self.hyperparams["n_epochs"]
        )

        del data_train
        if data_validation is not None:
            del data_validation
            
        result = {
            "train_losses": train_losses,
            "validation_losses": validation_losses,
        }
        
        return result

class Recommender_binary_logit_negative_sampling(Recommender_binary_logit):
    """Binary Logit Model (Matrix Factorization) with negative sampling"""
    def __init__(self, hyperparams=None):
        from src.models.binary_logit_negative_sampling import Recommender_Network
        self.hyperparams = hyperparams

        # Run on CPU, because its faster for this small data set size
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        
        self.model = Recommender_Network(
            n_users=hyperparams["n_users"], 
            n_items=hyperparams["n_alternatives"], 
            embedding_size=hyperparams["k"], 
            batch_size=hyperparams["batch_size"],
            l2_embs = float(10**hyperparams["l2_embs_log10"]),
            n_early_stop=hyperparams["n_early_stop"],
        )

    def train(self, data_train, data_validation=None, hyperparams=None):
        reload(src.models.binary_logit_negative_sampling)

        # Daten vorverarbeiten
        train_losses, validation_losses = self.model.train_steps(
            data_train=data_train,
            data_test=data_validation,
            learning_rate=self.hyperparams["learning_rate"],
            n_epochs=self.hyperparams["n_epochs"])

        del data_train
        if data_validation is not None:
            del data_validation

        result = {
            "train_losses": train_losses,
            "validation_losses": validation_losses,
        }
        return result

class Recommender_baselines(Recommender):
    """Base class for any recommenders that use a baselines repo."""
    def __init__(self, hyperparams=None):
        # Translate between naming conventions in our and the baseline repos
        hyperparams["hidden"] = hyperparams["k"]
        hyperparams["lam"] = float(10**hyperparams["l2_embs_log10"])
        hyperparams["max_iters"] = hyperparams["n_epochs"]
        hyperparams["eta"] = hyperparams["learning_rate"]

        hyperparams["data"] = 'DFKI'
        hyperparams["neg_sample"] = 3
        hyperparams["random_state"] = 1
        hyperparams["unbiased_eval"] = False
        
        hyperparams["alpha"] = hyperparams.get("alpha", 0)

        self.hyperparams = hyperparams
        
        if not hasattr(self, "required_hyperparamter_keys"):
            raise NotImplementedError()
        assert(np.all([key in hyperparams.keys() for key in self.required_hyperparamter_keys])), f"Missing hyperparameters {[key for key in self.required_hyperparamter_keys if key not in hyperparams.keys()]}"
        assert("model_name" in hyperparams.keys()), "'model_name' is missing"
        
        import tensorflow as tf

        tf.compat.v1.disable_eager_execution() # Added for compatibility with tf 2.8.0

        warnings.filterwarnings("ignore")
        tf.get_logger().setLevel("ERROR")

        # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        tf.config.set_visible_devices([], 'GPU') # disable GPU (not necessary here)

        if self.hyperparams["model_name"] == 'cjmf':
            self.batch_size = int(self.batch_size * 1. * (self.C - 1) / self.C)
    
    def train(self, data_train, data_validation=None, hyperparams=None):
        reload(sys.modules['src.models.Lee_2022_BISER.trainer'])
        from src.models.Lee_2022_BISER.trainer import Trainer

        from src.models.Lee_2022_BISER.util.preprocessor import preprocess_dataset
        train, val, pscore, item_freq = preprocess_dataset(data_train, data_validation, alpha=self.hyperparams["alpha"])

        self.trainer = Trainer(hyperparams=hyperparams)

        train_losses, validation_losses = self.trainer.run(train, val, pscore, item_freq)
        
        del data_train
        if data_validation is not None:
            del data_validation
            
        result = {
            "train_losses": train_losses,
            "validation_losses": validation_losses,
        }

        return result

    def predict(self, users, choiceset, rec_size=None):
        preds = []
        for user in users:
            utilities = self.trainer.predict(user, np.asarray(choiceset))
            pred_user = [x for _, x in sorted(zip(utilities, choiceset), reverse=True)]
            preds.append(pred_user)

        preds = np.asarray(preds)
        if rec_size is not None:
            preds = preds[:, range(rec_size)]
        return(preds)

    def clear(self):
            import tensorflow as tf
            tf.keras.backend.clear_session() 

class Recommender_biser(Recommender_baselines):
    """BISER"""
    def __init__(self, hyperparams=None):

        hyperparams["model_name"] = 'proposed'
        
        self.required_hyperparamter_keys = [
            "k",
            "learning_rate",
            "batch_size",
            "n_epochs",
            "l2_embs_log10",
            "wu",
            "wi",
            # "clip",
            # "alpha",
            # "C",
            # "alpha_cjmf",
            # "beta_cjmf",
            # "macr_c",
            # "macr_alpha",
            # "macr_beta",
        ]

        super(Recommender_biser, self).__init__(hyperparams)

class Recommender_relmf(Recommender_baselines):
    """RelMF"""
    def __init__(self, hyperparams=None):

        hyperparams["model_name"] = 'relmf'
        
        self.required_hyperparamter_keys = [
            "k",
            "learning_rate",
            "batch_size",
            "n_epochs",
            "l2_embs_log10",
            # "wu",
            # "wi",
            "clip",
            # "alpha",
            # "C",
            # "alpha_cjmf",
            # "beta_cjmf",
            # "macr_c",
            # "macr_alpha",
            # "macr_beta",
        ]

        super(Recommender_relmf, self).__init__(hyperparams)

class Recommender_macr(Recommender_baselines):
    """MACR"""
    def __init__(self, hyperparams=None):

        hyperparams["model_name"] = 'macr'
        
        self.required_hyperparamter_keys = [
            "k",
            "learning_rate",
            "batch_size",
            "n_epochs",
            "l2_embs_log10",
            # "wu",
            # "wi",
            # "clip",
            # "alpha",
            # "C",
            # "alpha_cjmf",
            # "beta_cjmf",
            "macr_c",
            "macr_alpha",
            "macr_beta",
        ]

        hyperparams["macr_alpha"] = float(10**hyperparams["macr_alpha_log10"])
        hyperparams["macr_beta"] = float(10**hyperparams["macr_beta_log10"])

        super(Recommender_macr, self).__init__(hyperparams)


class Recommender_pd(Recommender_baselines):
    """PD"""
    def __init__(self, hyperparams=None):

        hyperparams["model_name"] = 'pd'
        
        self.required_hyperparamter_keys = [
            "k",
            "learning_rate",
            "batch_size",
            "n_epochs",
            "l2_embs_log10",
            # "wu",
            # "wi",
            # "clip",
            "alpha",
            "gamma",
            # "C",
            # "alpha_cjmf",
            # "beta_cjmf",
            # "macr_c",
            # "macr_alpha",
            # "macr_beta",
        ]

        super(Recommender_pd, self).__init__(hyperparams)


class Recommender_bpr(Recommender_baselines):
    """Bayesian Personalized Ranking"""
    def __init__(self, hyperparams=None):

        hyperparams["model_name"] = 'bpr'
        
        self.required_hyperparamter_keys = [
            "k",
            "learning_rate",
            "batch_size",
            "n_epochs",
            "l2_embs_log10",
            # "wu",
            # "wi",
            # "clip",
            # "alpha",
            # "gamma",
            # "C",
            # "alpha_cjmf",
            # "beta_cjmf",
            # "macr_c",
            # "macr_alpha",
            # "macr_beta",
        ]

        super(Recommender_bpr, self).__init__(hyperparams)


class Recommender_ubpr(Recommender_baselines):
    """Unbiased Bayesian Personalized Ranking"""
    def __init__(self, hyperparams=None):

        hyperparams["model_name"] = 'ubpr'
        
        self.required_hyperparamter_keys = [
            "k",
            "learning_rate",
            "batch_size",
            "n_epochs",
            "l2_embs_log10",
            # "wu",
            # "wi",
            # "clip",
            # "alpha",
            # "gamma",
            # "C",
            # "alpha_cjmf",
            # "beta_cjmf",
            # "macr_c",
            # "macr_alpha",
            # "macr_beta",
            "ps_pow",
            "clip_min",
        ]

        super(Recommender_ubpr, self).__init__(hyperparams)


class Recommender_cpr(Recommender_baselines):
    """CPR"""
    def __init__(self, hyperparams=None):

        hyperparams["model_name"] = 'cpr'
        
        self.required_hyperparamter_keys = [
            "k",
            "learning_rate",
            "batch_size",
            "n_epochs",
            "l2_embs_log10",
            # "wu",
            # "wi",
            # "clip",
            # "alpha",
            # "gamma",
            # "C",
            # "alpha_cjmf",
            # "beta_cjmf",
            # "macr_c",
            # "macr_alpha",
            # "macr_beta",
            # "ps_pow",
            # "clip_min",
            "beta",
            "max_k_interact",
            "gamma",
            "n_step",
        ]

        super(Recommender_cpr, self).__init__(hyperparams)


class Recommender_dice(Recommender_baselines):
    """DICE"""
    def __init__(self, hyperparams=None):

        hyperparams["model_name"] = 'dice'
        
        self.required_hyperparamter_keys = [
            "k",
            "learning_rate",
            "batch_size",
            "n_epochs",
            "l2_embs_log10",
            # "wu",
            # "wi",
            # "clip",
            # "alpha",
            # "gamma",
            # "C",
            # "alpha_cjmf",
            # "beta_cjmf",
            # "macr_c",
            # "macr_alpha",
            # "macr_beta",
            # "ps_pow",
            # "clip_min",
            # "beta",
            # "max_k_interact",
            # "gamma",
            "init_margin",
            "dis_pen",
            "init_int_weight",
            "init_pop_weight",
            "loss_decay",
            "margin_decay",
            "pool",
        ]

        hyperparams["dis_pen"] = float(10**hyperparams["dis_pen_log10"])

        assert(hyperparams["k"] // 2 == hyperparams["k"] / 2), "k must be even"

        super(Recommender_dice, self).__init__(hyperparams)

        assert(hyperparams["pool"] >= hyperparams["neg_sample"]), "Pool must be larger than or equal to negative_sample_size."