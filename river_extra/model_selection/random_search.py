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

from scipy.stats import qmc
from tqdm import tqdm

import numpy as np

# TODO use lazy imports where needed
from river import anomaly, base, compose, drift, metrics, utils, preprocessing, tree

ModelWrapper = collections.namedtuple("ModelWrapper", "estimator metric")


class Random_Search(base.Estimator):

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

        self._counter=0
        self._best_estimator = None
        self._bandits = self._create_bandits(estimator,nr_estimators)

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
            return self._rng.randint(hp_range[0], hp_range[1])
        elif hp_type == float:
            return self._rng.uniform(hp_range[0], hp_range[1])

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
        return bandits

    def _sort_bandits(self):
        """Ensure the simplex models are ordered by predictive performance."""
        if self.metric.bigger_is_better:
            self._bandits.sort(key=lambda mw: mw.metric.get(), reverse=True)
        else:
            self._bandits.sort(key=lambda mw: mw.metric.get())

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

    def _learn_not_converged(self, x, y):
        for wrap in self._bandits:
            scorer = getattr(wrap.estimator, self._scorer_name)
            y_pred = scorer(x)
            wrap.metric.update(y, y_pred)
            wrap.estimator.learn_one(x, y)

        # Keep the simplex ordered


        if self._n == self.grace_period:
            self._sort_bandits()
            self._n = 0


    def learn_one(self, x, y):
        self._n += 1
        self._counter += 1

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
