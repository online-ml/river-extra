from ast import operator
import collections
import copy
import math
import numbers
import random
import typing

import numpy as np

# TODO use lazy imports where needed
from river import anomaly, base, compose, drift, metrics, utils

ModelWrapper = collections.namedtuple("ModelWrapper", "estimator metric")


class SSPT(base.Estimator):
    """Single-pass Self Parameter Tuning

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

        self.vtau=[]
        self.vdelta=[]
        self.vgrace=[]
        self.vscore=[]

        self.seed = seed

        self._n = 0
        self._converged = False
        self._rng = random.Random(self.seed)

        self._best_estimator = None
        self._simplex = self._create_simplex(estimator)

        # Models expanded from the simplex
        self._expanded: typing.Optional[typing.Dict] = None

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

    def __generate(self, p_data) -> numbers.Number:
        p_type, p_range = p_data
        if p_type == int:
            return self._rng.randint(p_range[0], p_range[1])
        elif p_type == float:
            return self._rng.uniform(p_range[0], p_range[1])

    def __combine(self, p_info, param1, param2, func):

        p_type, p_range = p_info
        new_val = func(param1, param2)

        # Range sanity checks
        if new_val < p_range[0]:
            new_val = p_range[0]
        if new_val > p_range[1]:
            new_val = p_range[1]

        new_val = round(new_val, 0) if p_type == int else new_val
        return new_val

    def __flatten(self, prefix, scaled, p_info, e_info):
        _, p_range = p_info
        interval = p_range[1] - p_range[0]
        scaled[prefix] = (e_info - p_range[0]) / interval

    def _recurse_params(
        self, operation, p_data, e1_data, *, func=None, e2_data=None, prefix=None, scaled=None
    ):
        # Sub-component needs to be instantiated
        if isinstance(e1_data, tuple):
            sub_class, sub_data1 = e1_data

            if operation == "combine":
                _, sub_data2 = e2_data
            else:
                sub_data2 = {}

            sub_config = {}
            for sub_param, sub_info in p_data.items():
                if operation == "scale":
                    sub_prefix = prefix + "__" + sub_param
                else:
                    sub_prefix = None

                sub_config[sub_param] = self._recurse_params(
                    operation=operation,
                    p_data=sub_info,
                    e1_data=sub_data1[sub_param],
                    func=func,
                    e2_data=sub_data2.get(sub_param, None),
                    prefix=sub_prefix,
                    scaled=scaled,
                )
            return sub_class(**sub_config)

        # We reached the numeric parameters
        if isinstance(p_data, tuple):
            if operation == "generate":
                return self.__generate(p_data)
            if operation == "scale":
                self.__flatten(prefix, scaled, p_data, e1_data)
                return
            # combine
            return self.__combine(p_data, e1_data, e2_data, func)

        # The sub-parameters need to be expanded
        config = {}
        for p_name, p_info in p_data.items():
            e1_info = e1_data[p_name]

            if operation == "combine":
                e2_info = e2_data[p_name]
            else:
                e2_info = {}

            if operation == "scale":
                sub_prefix = prefix + "__" + p_name if len(prefix) > 0 else p_name
            else:
                sub_prefix = None

            if not isinstance(p_info, dict):
                config[p_name] = self._recurse_params(
                    operation=operation,
                    p_data=p_info,
                    e1_data=e1_info,
                    func=func,
                    e2_data=e2_info,
                    prefix=sub_prefix,
                    scaled=scaled,
                )
            else:
                sub_config = {}
                for sub_name, sub_info in p_info.items():

                    if operation == "scale":
                        sub_prefix2 = sub_prefix + "__" + sub_name
                    else:
                        sub_prefix2 = None

                    sub_config[sub_name] = self._recurse_params(
                        operation=operation,
                        p_data=sub_info,
                        e1_data=e1_info[sub_name],
                        func=func,
                        e2_data=e2_info.get(sub_name, None),
                        prefix=sub_prefix2,
                        scaled=scaled,
                    )
                config[p_name] = sub_config
        return config

    def _random_config(self):
        return self._recurse_params(
            operation="generate",
            p_data=self.params_range,
            e1_data=self.estimator._get_params()
        )

    def _create_simplex(self, model) -> typing.List:
        # The simplex is divided in:
        # * 0: the best model
        # * 1: the 'good' model
        # * 2: the worst model
        simplex = [None] * 3

        simplex[0] = ModelWrapper(
            self.estimator.clone(self._random_config(), include_attributes=True),
            self.metric.clone(include_attributes=True),
        )
        simplex[1] = ModelWrapper(
            model.clone(self._random_config(), include_attributes=True),
            self.metric.clone(include_attributes=True),
        )
        simplex[2] = ModelWrapper(
            self.estimator.clone(self._random_config(), include_attributes=True),
            self.metric.clone(include_attributes=True),
        )

        return simplex

    def _sort_simplex(self):
        """Ensure the simplex models are ordered by predictive performance."""
        if self.metric.bigger_is_better:
            self._simplex.sort(key=lambda mw: mw.metric.get(), reverse=True)
        else:
            self._simplex.sort(key=lambda mw: mw.metric.get())

    def _gen_new_estimator(self, e1, e2, func):
        """Generate new configuration given two estimators and a combination function."""

        e1_p = e1.estimator._get_params()
        e2_p = e2.estimator._get_params()

        new_config = self._recurse_params(
            operation="combine",
            p_data=self.params_range,
            e1_data=e1_p,
            func=func,
            e2_data=e2_p
        )
        # Modify the current best contender with the new hyperparameter values
        new = ModelWrapper(
            copy.deepcopy(self._simplex[0].estimator),
            self.metric.clone(include_attributes=True),
        )
        new.estimator.mutate(new_config)

        return new

    def _nelder_mead_expansion(self) -> typing.Dict:
        """Create expanded models given the simplex models."""
        expanded = {}
        # Midpoint between 'best' and 'good'
        expanded["midpoint"] = self._gen_new_estimator(
            self._simplex[0], self._simplex[1], lambda h1, h2: (h1 + h2) / 2
        )

        # Reflection of 'midpoint' towards 'worst'
        expanded["reflection"] = self._gen_new_estimator(
            expanded["midpoint"], self._simplex[2], lambda h1, h2: 2 * h1 - h2
        )
        # Expand the 'reflection' point
        expanded["expansion"] = self._gen_new_estimator(
            expanded["reflection"], expanded["midpoint"], lambda h1, h2: 2 * h1 - h2
        )
        # Shrink 'best' and 'worst'
        expanded["shrink"] = self._gen_new_estimator(
            self._simplex[0], self._simplex[2], lambda h1, h2: (h1 + h2) / 2
        )
        # Contraction of 'midpoint' and 'worst'
        expanded["contraction1"] = self._gen_new_estimator(
            expanded["midpoint"], self._simplex[2], lambda h1, h2: (h1 + h2) / 2
        )
        # Contraction of 'midpoint' and 'reflection'
        expanded["contraction2"] = self._gen_new_estimator(
            expanded["midpoint"], expanded["reflection"], lambda h1, h2: (h1 + h2) / 2
        )

        return expanded

    def _nelder_mead_operators(self):
        b = self._simplex[0].metric
        g = self._simplex[1].metric
        w = self._simplex[2].metric
        r = self._expanded["reflection"].metric
        c1 = self._expanded["contraction1"].metric
        c2 = self._expanded["contraction2"].metric

        scaled_params = list(self._normalize_flattened_hyperspace(
            self._simplex[0].estimator._get_params(),
        ).values())
        self.vtau.append(scaled_params[0])
        self.vdelta.append(scaled_params[1])
        self.vgrace.append(scaled_params[2])
        self.vscore.append(self._simplex[0].metric.get())

        scaled_params = list(self._normalize_flattened_hyperspace(
            self._simplex[1].estimator._get_params(),
        ).values())
        self.vtau.append(scaled_params[0])
        self.vdelta.append(scaled_params[1])
        self.vgrace.append(scaled_params[2])
        self.vscore.append(self._simplex[1].metric.get())

        scaled_params = list(self._normalize_flattened_hyperspace(
            self._simplex[2].estimator._get_params(),
        ).values())
        self.vtau.append(scaled_params[0])
        self.vdelta.append(scaled_params[1])
        self.vgrace.append(scaled_params[2])
        self.vscore.append(self._simplex[2].metric.get())

        scaled_params = list(self._normalize_flattened_hyperspace(
            self._expanded["reflection"].estimator._get_params(),
        ).values())
        self.vtau.append(scaled_params[0])
        self.vdelta.append(scaled_params[1])
        self.vgrace.append(scaled_params[2])
        self.vscore.append(self._expanded["reflection"].metric.get())

        scaled_params = list(self._normalize_flattened_hyperspace(
            self._expanded["contraction1"].estimator._get_params(),
        ).values())
        self.vtau.append(scaled_params[0])
        self.vdelta.append(scaled_params[1])
        self.vgrace.append(scaled_params[2])
        self.vscore.append(self._expanded["contraction1"].metric.get())

        scaled_params = list(self._normalize_flattened_hyperspace(
            self._expanded["contraction2"].estimator._get_params(),
        ).values())
        self.vtau.append(scaled_params[0])
        self.vdelta.append(scaled_params[1])
        self.vgrace.append(scaled_params[2])
        self.vscore.append(self._expanded["contraction2"].metric.get())

        scaled_params = list(self._normalize_flattened_hyperspace(
            self._expanded["expansion"].estimator._get_params(),
        ).values())
        self.vtau.append(scaled_params[0])
        self.vdelta.append(scaled_params[1])
        self.vgrace.append(scaled_params[2])
        self.vscore.append(self._expanded["expansion"].metric.get())

        scaled_params = list(self._normalize_flattened_hyperspace(
            self._expanded["midpoint"].estimator._get_params(),
        ).values())
        self.vtau.append(scaled_params[0])
        self.vdelta.append(scaled_params[1])
        self.vgrace.append(scaled_params[2])
        self.vscore.append(self._expanded["midpoint"].metric.get())

        scaled_params = list(self._normalize_flattened_hyperspace(
            self._expanded["shrink"].estimator._get_params(),
        ).values())
        self.vtau.append(scaled_params[0])
        self.vdelta.append(scaled_params[1])
        self.vgrace.append(scaled_params[2])
        self.vscore.append(self._expanded["shrink"].metric.get())

        if c1.is_better_than(c2):
            self._expanded["contraction"] = self._expanded["contraction1"]
        else:
            self._expanded["contraction"] = self._expanded["contraction2"]
        if r.is_better_than(g):
            if b.is_better_than(r):
                self._simplex[2] = self._expanded["reflection"]
            else:
                e = self._expanded["expansion"].metric
                if e.is_better_than(b):
                    self._simplex[2] = self._expanded["expansion"]
                else:
                    self._simplex[2] = self._expanded["reflection"]
        else:
            if r.is_better_than(w):
                self._simplex[2] = self._expanded["reflection"]
            else:
                c = self._expanded["contraction"].metric
                if c.is_better_than(w):
                    self._simplex[2] = self._expanded["contraction"]
                else:
                    self._simplex[2] = self._expanded["shrink"]
                    self._simplex[1] = self._expanded["midpoint"]

        self._sort_simplex()

    def _normalize_flattened_hyperspace(self, orig):
        scaled = {}
        self._recurse_params(
            operation="scale",
            p_data=self.params_range,
            e1_data=orig,
            prefix="",
            scaled=scaled
        )
        return scaled

    @property
    def _models_converged(self) -> bool:
        # Normalize params to ensure they contribute equally to the stopping criterion

        # 1. Simplex in sphere
        scaled_params_b = self._normalize_flattened_hyperspace(
            self._simplex[0].estimator._get_params()
        )
        scaled_params_g = self._normalize_flattened_hyperspace(
            self._simplex[1].estimator._get_params()
        )
        scaled_params_w = self._normalize_flattened_hyperspace(
            self._simplex[2].estimator._get_params()
        )

        max_dist = max(
            [
                utils.math.minkowski_distance(scaled_params_b, scaled_params_g, p=2),
                utils.math.minkowski_distance(scaled_params_b, scaled_params_w, p=2),
                utils.math.minkowski_distance(scaled_params_g, scaled_params_w, p=2),
            ]
        )

        hyper_points = [
            list(scaled_params_b.values()),
            list(scaled_params_g.values()),
            list(scaled_params_w.values()),
        ]

        vectors = np.array(hyper_points)
        new_centroid = dict(zip(scaled_params_b.keys(), np.mean(vectors, axis=0)))
        centroid_distance = utils.math.minkowski_distance(
            self._old_centroid, new_centroid, p=2
        )
        self._old_centroid = new_centroid
        ndim = len(scaled_params_b)
        r_sphere = max_dist * math.sqrt((ndim / (2 * (ndim + 1))))

        if r_sphere < self.convergence_sphere or centroid_distance == 0:
            return True

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
            self._simplex = self._create_simplex(self._best_estimator)

            # There is no proven best model right now
            self._best_estimator = None
            return

        self._best_estimator.learn_one(x, y)

    def _learn_not_converged(self, x, y):
        for wrap in self._simplex:
            scorer = getattr(wrap.estimator, self._scorer_name)
            y_pred = scorer(x)
            wrap.metric.update(y, y_pred)
            wrap.estimator.learn_one(x, y)

        # Keep the simplex ordered
        self._sort_simplex()

        if not self._expanded:
            self._expanded = self._nelder_mead_expansion()

        for wrap in self._expanded.values():
            scorer = getattr(wrap.estimator, self._scorer_name)
            y_pred = scorer(x)
            wrap.metric.update(y, y_pred)
            wrap.estimator.learn_one(x, y)

        if self._n == self.grace_period:
            self._n = 0
            # 1. Simplex in sphere
            scaled_params_b = self._normalize_flattened_hyperspace(
                self._simplex[0].estimator._get_params(),
            )
            scaled_params_g = self._normalize_flattened_hyperspace(
                self._simplex[1].estimator._get_params(),
            )
            scaled_params_w = self._normalize_flattened_hyperspace(
                self._simplex[2].estimator._get_params(),
            )
            print("----------")
            print(
                "B:", list(scaled_params_b.values()), "Score:", self._simplex[0].metric
            )
            print(
                "G:", list(scaled_params_g.values()), "Score:", self._simplex[1].metric
            )
            print(
                "W:", list(scaled_params_w.values()), "Score:", self._simplex[2].metric
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

            # Update the simplex models using Nelder-Mead heuristics
            self._nelder_mead_operators()

            # Discard expanded models
            self._expanded = None

            if self._models_converged:
                self._converged = True
                self._best_estimator = self._simplex[0].estimator

    def learn_one(self, x, y):
        self._n += 1

        if self.converged:
            self._learn_converged(x, y)
        else:
            self._learn_not_converged(x, y)

        return self

    @property
    def best(self):
        if not self._converged:
            # Lazy selection of the best model
            self._sort_simplex()
            return self._simplex[0].estimator

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
            return self.best.score_one(x, **kwargs)
        except NotImplementedError:
            raise AttributeError(
                f"'debug_one' is not supported in {self.best.__class__.__name__}."
            )
