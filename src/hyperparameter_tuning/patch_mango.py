import numpy as np
from tqdm import tqdm
import mango
import random
import math
from sklearn.cluster import KMeans
from mango.optimizer.bayesian_learning import BayesianLearning

def patch_mango():
    """Monkey-patches mango to fix some bugs and add some features."""
    # Monkey-patch the BayesianLearner to use a proper alpha value
    def new_init(self, surrogate=None, alpha=None, domain_size=1000):
        # initialzing some of the default values
        # The default surrogate function is gaussian_process with matern kernel
        if surrogate is None:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern
            self.surrogate = GaussianProcessRegressor(kernel=Matern(nu=2.5),
                                                        n_restarts_optimizer=10,
                                                        # FIXME:check if we should be passing this
                                                        # random state
                                                        random_state=1,
                                                        normalize_y=True,
                                                        alpha=1,
                                                        )
        else:
            self.surrogate = surrogate

        # keep track of the iteration counts
        self.iteration_count = 0

        # The size of the exploration domain, default to 1000
        self.domain_size = domain_size

        self.alpha = alpha
    mango.optimizer.bayesian_learning.BayesianLearning.__init__ = new_init

    # Monkey-patch the early stopping function to include info about the optimizer
    def new_early_stop(self, results, optimizer, ds, X_sample, pbar):
        if self.early_stopping is None:
            return False,

        # results = copy.deepcopy(results)
        return self.early_stopping(results, optimizer, ds, X_sample, pbar)
    mango.Tuner.Config.early_stop = new_early_stop

    # Monkey-patch the Bayesian Optimizer, mainly to adjust to the early stopping patch
    def new_runBayesianOptimizer(self):
        results = dict()

        X_list, Y_list, X_tried = self.run_initial() # We do not need X_tried anymore. We use list(X_list), because its the same, but accounts for CV

        # evaluated hyper parameters are used
        X_init = self.ds.convert_GP_space(X_list)
        Y_init = Y_list.reshape(len(Y_list), 1)

        # setting the initial random hyper parameters tried
        results["random_params"] = X_list
        results["random_params_objective"] = Y_list

        Optimizer = BayesianLearning(
            surrogate=self.config.surrogate,
            alpha=self.config.alpha,
            domain_size=self.config.domain_size,
        )

        X_sample = X_init
        Y_sample = Y_init

        hyper_parameters_tried = list(X_list)
        objective_function_values = Y_list
        surrogate_values = Y_list

        x_failed_evaluations = np.array([])

        # domain_list = self.ds.get_domain()
        # X_domain_np = self.ds.convert_GP_space(domain_list)
        context = None

        # running the iterations
        pbar = tqdm(range(self.config.num_iteration), ascii=True, leave=False)
        for i in pbar:

            # Moved these lines here.
            # The domain_list is a random sample of hyperparameters used for determining the next evaluation batch
            # We re-draw this list in every iteration for less overfitting on the domain_list
            domain_list = self.ds.get_domain()
            X_domain_np = self.ds.convert_GP_space(domain_list)

            # adding a Minimum exploration to explore independent of UCB
            if random.random() < self.config.exploration:
                random_parameters = self.ds.get_random_sample(self.config.batch_size)
                X_next_batch = self.ds.convert_GP_space(random_parameters)

                if self.config.exploration > self.config.exploration_min:
                    self.config.exploration = (
                        self.config.exploration * self.config.exploration_decay
                    )

            elif self.config.strategy_is_penalty:
                X_next_batch = Optimizer.get_next_batch(
                    X_sample, Y_sample, X_domain_np, batch_size=self.config.batch_size
                )
            elif self.config.strategy_is_clustering:
                X_next_batch = Optimizer.get_next_batch_clustering(
                    X_sample, Y_sample, X_domain_np, batch_size=self.config.batch_size
                )
            else:
                # assume penalty approach
                X_next_batch = Optimizer.get_next_batch(
                    X_sample, Y_sample, X_domain_np, batch_size=self.config.batch_size
                )

            # Scheduler
            X_next_PS = self.ds.convert_PS_space(X_next_batch)

            # if all the xs have failed before, replace them with random sample
            # as we will not get any new information otherwise
            if all(x in x_failed_evaluations for x in X_next_PS):
                X_next_PS = self.ds.get_random_sample(self.config.batch_size)

            # Evaluate the Objective function
            X_next_list, Y_next_list = self.runUserObjective(X_next_PS)

            # keep track of all parameters that failed
            x_failed = [x for x in X_next_PS if x not in X_next_list]
            x_failed_evaluations = np.append(x_failed_evaluations, x_failed)

            if len(Y_next_list) == 0:
                # no values returned
                # this is problematic if domain is small and same value is tried again in the next iteration as the optimizer would be stuck
                continue

            Y_next_batch = Y_next_list.reshape(len(Y_next_list), 1)
            # update X_next_batch to successfully evaluated values
            X_next_batch = self.ds.convert_GP_space(X_next_list)

            # update the bookeeping of values tried
            hyper_parameters_tried = np.append(hyper_parameters_tried, X_next_list)
            objective_function_values = np.append(
                objective_function_values, Y_next_list
            )
            surrogate_values = np.append(
                surrogate_values, Optimizer.surrogate.predict(X_next_batch)
            )

            # Appending to the current samples
            X_sample = np.vstack((X_sample, X_next_batch))
            Y_sample = np.vstack((Y_sample, Y_next_batch))

            # referesh domain if not fixed
            if not self.config.fixed_domain:
                domain_list = self.ds.get_domain()
                X_domain_np = self.ds.convert_GP_space(domain_list)

            results["params_tried"] = hyper_parameters_tried
            results["objective_values"] = objective_function_values
            results["surrogate_values"] = surrogate_values

            results["best_objective"] = np.max(results["objective_values"])
            results["best_params"] = results["params_tried"][
                np.argmax(np.reshape(Optimizer.Get_Lower_Confidence_Bound(X_sample), -1)) # base on the current "best" value of the surrogate function; the objective_values are too noisy for determining the best parameters
            ]
            if self.maximize_objective is False:
                results["objective_values"] = -1 * results["objective_values"]
                results["best_objective"] = -1 * results["best_objective"]

            # pbar.set_description("Best score: %s" % results["best_objective"])

            # Alpha can "be interpreted as the variance of additional Gaussian measurement noise on the training observations" (After normalizing the input data)
            # Therefore, we set it to the variance of the standardized objective values corresponding to the currently best hyperparameters
            # To control for variance, we use a running mean
            # See https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
            if len(results["objective_values"]) % 3 == 0: # If we are doing CV (If not, Optimizer.local_variance does not matter anyways)
                Optimizer.local_variance = np.mean(
                    np.reshape(results["objective_values"], (len(results["objective_values"])//3, 3)).std(axis=1)**2
                    )
            else:
                Optimizer.local_variance = 1
            new_alpha = Optimizer.local_variance / (results["objective_values"].std()**2)
            # new_alpha = (
            #     np.std((results["objective_values"][np.where(results["params_tried"] == results["best_params"])] 
            #     - results["objective_values"].mean())/results["objective_values"].std())
            # )**2
            Optimizer.surrogate.alpha = new_alpha
            # print(f"Alpha: {Optimizer.surrogate.alpha}")
            # (np.std((results["objective_values"][np.where(results["params_tried"] == results["best_params"])] - results["objective_values"].mean())/results["objective_values"].std()))**2
            # (np.std(
            #     results["objective_values"][np.where(results["params_tried"] == results["best_params"])]
            #     )/np.std(results["objective_values"]))**2


            # check if early stop criteria has been met
            if self.config.early_stop(results, Optimizer, self.ds, X_sample, pbar):
                # _logger.info("Early stopping criteria satisfied")
                break

        # saving the optimizer and ds in the tuner object which can save the surrogate function and ds details
        self.Optimizer = Optimizer

        return results
    mango.Tuner.runBayesianOptimizer = new_runBayesianOptimizer

    # Add a function to calculate the lower confidence bound for the early stopping
    def Get_Lower_Confidence_Bound(self, X):
        """
            Returns the acqutition function
        """
        mu, sigma = self.surrogate.predict(X, return_std=True)
        mu = mu.reshape(mu.shape[0], 1)
        sigma = sigma.reshape(sigma.shape[0], 1)

        if self.alpha is not None:
            exploration_factor = self.alpha
        else:
            alpha_inter = self.domain_size * (self.iteration_count) * (self.iteration_count) * math.pi * math.pi / (
                    6 * 0.1)
            if alpha_inter == 0:
                raise ValueError('alpha_inter is zero in Upper_Confidence_Bound')
            alpha = 2 * math.log(alpha_inter)  # We have set delta = 0.1
            alpha = math.sqrt(alpha)

            exploration_factor = alpha

        Value = mu - exploration_factor * sigma

        return Value
    mango.optimizer.bayesian_learning.BayesianLearning.Get_Lower_Confidence_Bound = Get_Lower_Confidence_Bound

    # Always pick the hyperparameters that maximize the Acquition function; do not filter out hyperparameters that are close to previously tried ones
    def new_get_next_batch_clustering(self, X, Y, X_tries, batch_size):
        # print('In get_next_batch')

        X_temp = X
        Y_temp = Y

        self.surrogate.fit(X_temp, Y_temp)
        self.iteration_count = self.iteration_count + 1

        Acquition = self.Get_Upper_Confidence_Bound(X_tries)

        if batch_size > 1:
            gen = sorted(zip(Acquition, X_tries), key=lambda x: -x[0])
            x_best_acq_value, x_best_acq_domain = (np.array(t)[:len(Acquition) // 4]
                                                    for t in zip(*gen))

            # Do the domain space based clustering on the best points
            kmeans = KMeans(n_clusters=batch_size, random_state=0).fit(x_best_acq_domain)
            cluster_pred_domain = kmeans.labels_.reshape(kmeans.labels_.shape[0])

            # Partition the space into the cluster in X and select the best X from each space
            partitioned_space = dict()
            partitioned_acq = dict()
            for i in range(batch_size):
                partitioned_space[i] = []
                partitioned_acq[i] = []

            for i in range(x_best_acq_domain.shape[0]):
                partitioned_space[cluster_pred_domain[i]].append(x_best_acq_domain[i])
                partitioned_acq[cluster_pred_domain[i]].append(x_best_acq_value[i])

            # Handle empty clusters
            for i in range(batch_size):
                if len(partitioned_space[i]) == 0:
                    # Assign random point
                    random_index = np.random.choice(len(x_best_acq_domain), 1, replace=False)[0]  
                    partitioned_space[i].append(x_best_acq_domain[random_index])
                    partitioned_acq[i].append(x_best_acq_value[random_index])

            batch = []

            for i in partitioned_space:
                x_local = partitioned_space[i]
                acq_local = partitioned_acq[i]
                acq_local = np.array(acq_local)
                x_index = np.argmax(acq_local)
                x_final_selected = x_local[x_index]
                batch.append([x_final_selected])

        else:  # batch_size ==1
            batch = []
            x_index = np.argmax(Acquition)
            # x_final_selected = self.remove_duplicates_serial(X_tries, X_temp, Acquition)
            x_final_selected = X_tries[x_index]
            batch.append([x_final_selected])

        batch = np.array(batch)
        batch = batch.reshape(-1, X.shape[1])
        return batch
    mango.optimizer.bayesian_learning.BayesianLearning.get_next_batch_clustering = new_get_next_batch_clustering

    # Handle multiple results (CV) per user objective run
    def new_runUserObjective(self, X_next_PS):
            # initially assuming entire X_next_PS is evaluated and returned results are only Y values
            X_list_evaluated = X_next_PS
            results = self.objective_function(X_next_PS)
            Y_list_evaluated = results

            # if result is a tuple, then there is possibility that partial values are evaluated
            if len(X_next_PS) > 0:
                if isinstance(results[0], tuple):
                    X_list_evaluated = [hp for r in results for hp in r[0]]
                    Y_list_evaluated = [score for r in results for score in r[1]]
            else:
                if isinstance(results, tuple):
                    X_list_evaluated = results[0]
                    Y_list_evaluated = results[1]

            X_list_evaluated = np.array(X_list_evaluated)
            Y_list_evaluated = np.array(Y_list_evaluated)
            if self.maximize_objective is False:
                Y_list_evaluated = -1 * Y_list_evaluated
            else:
                Y_list_evaluated = Y_list_evaluated

            return X_list_evaluated, Y_list_evaluated
    mango.Tuner.runUserObjective = new_runUserObjective
