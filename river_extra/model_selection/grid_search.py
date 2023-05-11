import statistics
from ast import operator
import collections
import copy
import math
import numbers
import random
import typing
import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm

import numpy as np

# TODO use lazy imports where needed
from river import anomaly, base, compose, drift, metrics, utils, preprocessing, tree

ModelWrapper = collections.namedtuple("ModelWrapper", "estimator metric")


class Grid_Search(base.Estimator):
    """Bandit Self Parameter Tuning

    Parameters
    ----------
    estimator
    metric
    params_range
    drift_input
    grace_period
    drift_detector
    convergence_sphere
    seed

    References
    ----------
    [1]: Veloso, B., Gama, J., Malheiro, B., & Vinagre, J. (2021). Hyperparameter self-tuning
    for data streams. Information Fusion, 76, 75-86.

    """

    _START_RANDOM = "random"
    _START_WARM = "warm"

    def __init__(
            self,
            estimator: base.Estimator,
            metric: metrics.base.Metric,
            params_range: typing.Dict[str, typing.Tuple],
            drift_input: typing.Callable[[float, float], float],
            grace_period: int = 500,
            drift_detector: base.DriftDetector = drift.ADWIN(),
            convergence_sphere: float = 0.001,
            nr_estimators: int = 50,
            seed: int = None,
    ):
        super().__init__()
        self.estimator = estimator
        self.metric = metric
        self.params_range = params_range
        self.drift_input = drift_input

        self.grace_period = grace_period
        self.drift_detector = drift_detector
        self.convergence_sphere = convergence_sphere

        self.nr_estimators = nr_estimators

        self.seed = seed

        self._n = 0
        self._converged = False
        self._rng = random.Random(self.seed)
        grids = np.meshgrid(np.linspace(0,1,len(params_range[self.estimator[1].__class__.__name__])), np.linspace(0,1,len(params_range[self.estimator[1].__class__.__name__])), np.linspace(0,1,len(params_range[self.estimator[1].__class__.__name__])), np.linspace(0,1,len(params_range[self.estimator[1].__class__.__name__])))
        self._sample=np.moveaxis(np.array(grids), 0, grids[0].ndim).reshape(-1, len(grids))

        self._i=0
        self._p=0
        self._counter=0
        self._best_estimator = None
        self._bandits = self._create_bandits(estimator,nr_estimators)

        # Convergence criterion
        self._old_centroid = None

        # Meta-programming
        border = self.estimator
        if isinstance(border, compose.Pipeline):
            border = border[-1]

        if isinstance(border, (base.Classifier, base.Regressor)):
            self._scorer_name = "predict_one"
        elif isinstance(border, anomaly.base.AnomalyDetector):
            self._scorer_name = "score_one"
        elif isinstance(border, anomaly.base.AnomalyDetector):
            self._scorer_name = "classify"

    def __generate(self, hp_data) -> numbers.Number:
        hp_type, hp_range = hp_data
        if hp_type == int:
            val = self._sample[self._i][self._p] * (hp_range[1] - hp_range[0]) + hp_range[0]
            self._p += 1
            return int (val)
        elif hp_type == float:
            val = self._sample[self._i][self._p] * (hp_range[1] - hp_range[0]) + hp_range[0]
            self._p += 1
            return val

    def __combine(self, hp_data, hp_est_1, hp_est_2, func) -> numbers.Number:
        hp_type, hp_range = hp_data
        new_val = func(hp_est_1, hp_est_2)

        # Range sanity checks
        if new_val < hp_range[0]:
            new_val = hp_range[0]
        if new_val > hp_range[1]:
            new_val = hp_range[1]

        new_val = round(new_val, 0) if hp_type == int else new_val
        return new_val

    def __combine_bee(self, hp_data, hp_est_1, hp_est_2, func) -> numbers.Number:
        hp_type, hp_range = hp_data
        if hp_est_1>=hp_est_2:
            func=lambda h1, h2: h2 + ((h1 + h2)*9 / 10)
        else:
            func = lambda h1, h2: h1 + ((h1 + h2)*9 / 10)
        new_val = func(hp_est_1, hp_est_2)

        # Range sanity checks
        if new_val < hp_range[0]:
            new_val = hp_range[0]
        if new_val > hp_range[1]:
            new_val = hp_range[1]

        new_val = round(new_val, 0) if hp_type == int else new_val
        return new_val

    def analyze_fanova(self, cal_df, max_iter=-1):
        metric='score'
        params = list(eval(cal_df.loc[0, 'params']).keys())
        result = pd.DataFrame(cal_df.loc[:, metric].copy())
        tmp_df = cal_df.loc[:, 'params'].copy()
        for key in params:
            result.insert(loc=0, column=key, value=tmp_df.apply(lambda x: eval(x)[key]))
        col_name = params[:]
        col_name.append(metric)
        cal_df = result.reindex(columns=col_name).copy()

        axis = ['id']
        axis.extend(list(cal_df.columns))
        params = axis[1:-1]
        metric = axis[-1]
        f = pd.DataFrame(columns=axis)
        f.loc[0, :] = np.nan
        f.loc[0, metric] = cal_df[metric].mean()
        f.loc[0, 'id'] = hash(str([]))
        v_all = np.std(cal_df[metric].to_numpy()) ** 2
        v = pd.DataFrame(columns=['u', 'v_u', 'F_u(v_u/v_all)'])
        for k in range(1, len(params) + 1):
            for u in combinations(params, k):
                ne_u = set(params) - set(u)
                a_u = cal_df.groupby(list(u)).mean().reset_index()
                if len(ne_u):
                    for nu in ne_u:
                        a_u.loc[:, nu] = np.nan
                col_name = cal_df.columns.tolist()
                a_u = a_u.reindex(columns=col_name)
                sum_f_w = pd.DataFrame(columns=f.columns[:])
                tmp = []
                w_list = []
                for i in range(len(u)):
                    tmp.extend(list(combinations(u, i)))
                for t in tmp:
                    w_list.append(list(t))

                for w in w_list:
                    sum_f_w = pd.concat([sum_f_w, f[f['id'] == hash(str(w))]], ignore_index=True)
                col_name = sum_f_w.columns.tolist()
                a_u = a_u.reindex(columns=col_name)
                for row_index, r in sum_f_w.iterrows():
                    r2 = r[1:-1]
                    not_null_index = r2.notnull().values
                    if not not_null_index.any():
                        a_u.loc[:, col_name[-1]] -= r[-1]
                    else:
                        left, right = a_u.iloc[:, 1:-1].align(r2[not_null_index], axis=1, copy=False)
                        equal_index = (left == right).values.sum(axis=1)
                        a_u.loc[equal_index == not_null_index.sum(), col_name[-1]] -= r[-1]
                a_u['id'] = hash(str(list(u)))
                f = pd.concat([f, a_u], ignore_index=True)
                tmp_f_u = a_u.loc[:, metric].to_numpy()
                v = pd.concat([v, pd.DataFrame(
                    [{'u': u, 'v_u': (tmp_f_u ** 2).mean(), 'F_u(v_u/v_all)': (tmp_f_u ** 2).mean() / v_all}])],
                              ignore_index=True)

        return v


    def __flatten(self, prefix, scaled_hps, hp_data, est_data):
        hp_range = hp_data[1]
        interval = hp_range[1] - hp_range[0]
        scaled_hps[prefix] = (est_data - hp_range[0]) / interval

    def _traverse_hps(
            self, operation: str, hp_data: dict, est_1, *, func=None, est_2=None, hp_prefix=None, scaled_hps=None
    ) -> typing.Optional[typing.Union[dict, numbers.Number]]:
        """Traverse the hyperparameters of the estimator/pipeline and perform an operation.

        Parameters
        ----------
        operation
            The operation that is intented to apply over the hyperparameters. Can be either:
            "combine" (combine parameters from two pipelines), "scale" (scale a flattened
            version of the hyperparameter hierarchy to use in the stopping criteria), or
            "generate" (create a new hyperparameter set candidate).
        hp_data
            The hyperparameter data which was passed by the user. Defines the ranges to
            explore for each hyperparameter.
        est_1
            The hyperparameter structure of the first pipeline/estimator. Such structure is obtained
            via a `_get_params()` method call. Both 'hp_data' and 'est_1' will be jointly traversed.
        func
            A function that is used to combine the values in `est_1` and `est_2`, in case
            `operation="combine"`.
        est_2
            A second pipeline/estimator which is going to be combined with `est_1`, in case
            `operation="combine"`.
        hp_prefix
            A hyperparameter prefix which is used to identify each hyperparameter in the hyperparameter
            hierarchy when `operation="scale"`. The recursive traversal will modify this prefix accordingly
            to the current position in the hierarchy. Initially it is set to `None`.
        scaled_hps
            Flattened version of the hyperparameter hierarchy which is used for evaluating stopping criteria.
            Set to `None` and defined automatically when `operation="scale"`.
        """

        # Sub-component needs to be instantiated
        if isinstance(est_1, tuple):
            sub_class, est_1 = est_1

            if operation == "combine":
                est_2 = est_2[1]
            else:
                est_2 = {}

            sub_config = {}
            for sub_hp_name, sub_hp_data in hp_data.items():
                if operation == "scale":
                    sub_hp_prefix = hp_prefix + "__" + sub_hp_name
                else:
                    sub_hp_prefix = None

                sub_config[sub_hp_name] = self._traverse_hps(
                    operation=operation,
                    hp_data=sub_hp_data,
                    est_1=est_1[sub_hp_name],
                    func=func,
                    est_2=est_2.get(sub_hp_name, None),
                    hp_prefix=sub_hp_prefix,
                    scaled_hps=scaled_hps,
                )
            return sub_class(**sub_config)

        # We reached the numeric parameters
        if isinstance(est_1, numbers.Number):
            if operation == "generate":
                return self.__generate(hp_data)
            if operation == "scale":
                self.__flatten(hp_prefix, scaled_hps, hp_data, est_1)
                return
            # combine
            return self.__combine(hp_data, est_1, est_2, func)

        # The sub-parameters need to be expanded
        config = {}
        for sub_hp_name, sub_hp_data in hp_data.items():
            sub_est_1 = est_1[sub_hp_name]

            if operation == "combine":
                sub_est_2 = est_2[sub_hp_name]
            else:
                sub_est_2 = {}

            if operation == "scale":
                sub_hp_prefix = hp_prefix + "__" + sub_hp_name if len(hp_prefix) > 0 else sub_hp_name
            else:
                sub_hp_prefix = None

            config[sub_hp_name] = self._traverse_hps(
                operation=operation,
                hp_data=sub_hp_data,
                est_1=sub_est_1,
                func=func,
                est_2=sub_est_2,
                hp_prefix=sub_hp_prefix,
                scaled_hps=scaled_hps,
            )

        return config

    def _random_config(self):
        return self._traverse_hps(
            operation="generate",
            hp_data=self.params_range,
            est_1=self.estimator._get_params(),
        )

    def _create_bandits(self, model, nr_estimators) -> typing.List:
        bandits = [None] * nr_estimators

        for i in range(nr_estimators):
            bandits[i] = ModelWrapper(
                model.clone(self._random_config(), include_attributes=True),
                self.metric.clone(include_attributes=True),
            )
            self._i+=1
            self._p=0
        self._i=0
        return bandits

    def _sort_bandits(self):
        """Ensure the simplex models are ordered by predictive performance."""
        if self.metric.bigger_is_better:
            self._bandits.sort(key=lambda mw: mw.metric.get(), reverse=True)
        else:
            self._bandits.sort(key=lambda mw: mw.metric.get())

    def _prune_bandits(self):
        """Ensure the simplex models are ordered by predictive performance."""
        self._bandits[0].estimator._get_params()
        half = int(1*len(self._bandits) / 10)
        for i in range(len(self._bandits)):
            #if i>half:
            lst=list(self.params_range[self.estimator[1].__class__.__name__].keys())
            mydict={}
            for x in lst:
                mydict[x]=self._bandits[i].estimator._get_params()[self.estimator[1].__class__.__name__][x]
                #mydict['instances']
            self._df_hpi.loc[len(self._df_hpi.index)] = [str(mydict), self._bandits[i].estimator[self.estimator[1].__class__.__name__].summary['total_observed_weight'], self._bandits[i].metric.get()]
                #'{\'hp1\': ' + str(row[1]['hp1']) + ', \'hp2\': ' + str(row[1]['hp2']) + ', \'hp3\': ' + str(row[1]['hp3']) + '}'
                #new_col = []
                #for row in df.iterrows():
                #    new_col.append(
                #        )
                #df['params'] = new_col
            self._pruned_configurations.append(str(self._bandits[i].estimator._get_params()['HoeffdingTreeClassifier']['delta'])+','+
                                                   str(self._bandits[i].estimator._get_params()['HoeffdingTreeClassifier']['tau'])+','+
                                                   str(self._bandits[i].estimator._get_params()['HoeffdingTreeClassifier']['grace_period'])+','+
                                                   str(self._bandits[i].estimator['HoeffdingTreeClassifier'].summary['total_observed_weight'])+','+str(self._bandits[i].metric.get()))
            #else:
                #self._pruned_configurations.append(
                #    str(self._bandits[i].estimator._get_params()['HoeffdingTreeClassifier']['delta']) + ',' +
                #    str(self._bandits[i].estimator._get_params()['HoeffdingTreeClassifier']['tau']) + ',' +
                #    str(self._bandits[i].estimator._get_params()['HoeffdingTreeClassifier'][
                #            'grace_period']) + ',' +
                #    str(self._bandits[i].estimator['HoeffdingTreeClassifier'].summary[
                #            'total_observed_weight']) + ',' + str(self._bandits[i].metric.get()))

        #if len(self._bandits)-half>2:
        #    self._bandits=self._bandits[:-half]
        #else:
        #    self._bandits=self._bandits[:-(len(self._bandits)-3)]


    def _gen_new_estimator(self, e1, e2, func):
        """Generate new configuration given two estimators and a combination function."""

        est_1_hps = e1.estimator._get_params()
        est_2_hps = e2.estimator._get_params()

        new_config = self._traverse_hps(
            operation="combine",
            hp_data=self.params_range,
            est_1=est_1_hps,
            func=func,
            est_2=est_2_hps
        )

        # Modify the current best contender with the new hyperparameter values
        new = ModelWrapper(
            copy.deepcopy(self._bandits[0].estimator),
            self.metric.clone(include_attributes=True),
        )
        new.estimator.mutate(new_config)

        return new


    def _normalize_flattened_hyperspace(self, orig):
        scaled = {}
        self._traverse_hps(
            operation="scale",
            hp_data=self.params_range,
            est_1=orig,
            hp_prefix="",
            scaled_hps=scaled
        )
        return scaled

    @property
    def _models_converged(self) -> bool:
        if len(self._bandits)==3:
            return True
        else:
            return False

    def _learn_converged(self, x, y):
        scorer = getattr(self._best_estimator, self._scorer_name)
        y_pred = scorer(x)

        input = self.drift_input(y, y_pred)
        self.drift_detector.update(input)

        # We need to start the optimization process from scratch
        if self.drift_detector.drift_detected:
            self._n = 0
            self._converged = False
            self._bandits = self._create_bandits(self._best_estimator, self.nr_estimators)

            # There is no proven best model right now
            self._best_estimator = None
            return

        self._best_estimator.learn_one(x, y)

    def _learn_not_converged(self, x, y):
        for wrap in self._bandits:
            scorer = getattr(wrap.estimator, self._scorer_name)
            y_pred = scorer(x)
            wrap.metric.update(y, y_pred)
            wrap.estimator.learn_one(x, y)

        # Keep the simplex ordered
        self._sort_bandits()

        if self._n == self.grace_period:
            self._n = 0
            self._prune_bandits()
            df3 = self._df_hpi[self._df_hpi['instances'] ==self._counter]
            df3 = df3.reset_index()
            importance=self.analyze_fanova(df3)
            print(importance)
            #hpi=importance[importance['F_u(v_u/v_all)']==max(importance['F_u(v_u/v_all)'])]['u'][0]

            #importance[importance['F_u(v_u/v_all)'] == max(importance['F_u(v_u/v_all)'])]
            print("Nr bandits: ",len(self._bandits))
            # 1. Simplex in sphere
            scaled_params_b = self._normalize_flattened_hyperspace(
                self._bandits[0].estimator._get_params(),
            )
            scaled_params_g = self._normalize_flattened_hyperspace(
                self._bandits[1].estimator._get_params(),
            )
            scaled_params_w = self._normalize_flattened_hyperspace(
                self._bandits[2].estimator._get_params(),
            )
            #for i in range(1,len(self._bandits)):
            #    self._bandits[i]=self._gen_new_estimator(self._bandits[0],self._bandits[i],lambda h1, h2: ((h1 + h2)*3) / 4)

            print("----------")
            print(
                "B:", list(scaled_params_b.values()), "Score:", self._bandits[0].metric
            )
            print(
                "G:", list(scaled_params_g.values()), "Score:", self._bandits[1].metric
            )
            print(
                "W:", list(scaled_params_w.values()), "Score:", self._bandits[2].metric
            )
            hyper_points = [
                list(scaled_params_b.values()),
                list(scaled_params_g.values()),
                list(scaled_params_w.values()),
            ]
            vectors = np.array(hyper_points)
            self._old_centroid = dict(
                zip(scaled_params_b.keys(), np.mean(vectors, axis=0))
            )


            if self._models_converged:
                self._converged = True
                self._best_estimator = self._bandits[0].estimator

    def learn_one(self, x, y):
        self._n += 1
        self._counter += 1

        if self.converged:
            self._learn_converged(x, y)
        else:
            self._learn_not_converged(x, y)

        return self

    @property
    def best(self):
        if not self._converged:
            # Lazy selection of the best model
            self._sort_bandits()
            return self._bandits[0].estimator

        return self._best_estimator

    @property
    def converged(self):
        return self._converged

    def predict_one(self, x, **kwargs):
        try:
            return self.best.predict_one(x, **kwargs)
        except NotImplementedError:
            border = self.best
            if isinstance(border, compose.Pipeline):
                border = border[-1]
            raise AttributeError(
                f"'predict_one' is not supported in {border.__class__.__name__}."
            )

    def predict_proba_one(self, x, **kwargs):
        try:
            return self.best.predict_proba_one(x, **kwargs)
        except NotImplementedError:
            border = self.best
            if isinstance(border, compose.Pipeline):
                border = border[-1]
            raise AttributeError(
                f"'predict_proba_one' is not supported in {border.__class__.__name__}."
            )

    def score_one(self, x, **kwargs):
        try:
            return self.best.score_one(x, **kwargs)
        except NotImplementedError:
            border = self.best
            if isinstance(border, compose.Pipeline):
                border = border[-1]
            raise AttributeError(
                f"'score_one' is not supported in {border.__class__.__name__}."
            )

    def debug_one(self, x, **kwargs):
        try:
            return self.best.debug_one(x, **kwargs)
        except NotImplementedError:
            raise AttributeError(
                f"'debug_one' is not supported in {self.best.__class__.__name__}."
            )
