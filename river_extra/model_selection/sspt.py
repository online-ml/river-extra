import abc
import collections
import copy
import math
import random
import types
import typing

from river import base, drift, metrics, utils

ModelWrapper = collections.namedtuple("ModelWrapper", "model metric")


class SSPT(base.Estimator):
    """Single-pass Self Parameter Tuning"""

    _START_RANDOM = "random"
    _START_WARM = "warm"

    def __init__(
        self,
        model,
        metric,
        params_range: typing.Dict[str, typing.Tuple],
        grace_period: int,
        drift_detector: base.DriftDetector,
        start: str,
        convergence_sphere: float,
        seed: int,
    ):
        super().__init__()
        self.model = model
        self.metric = metric
        self.params_range = params_range

        self.grace_period = grace_period
        self.drift_detector = drift_detector

        if start not in {self._START_RANDOM, self._START_WARM}:
            raise ValueError(
                f"'start' must be either '{self._START_RANDOM}' or '{self._START_WARM}'."
            )
        self.start = start
        self.convergence_sphere = convergence_sphere

        self.seed = seed

        self._n = 0
        self._converged = False
        self._rng = random.Random(self.seed)

        self._best_model = None
        self._simplex = self._create_simplex(model)

        # Models expanded from the simplex
        self._expanded: typing.Optional[typing.Dict] = None

        # Meta-programming
        self._bind_output_method()

    def _bind_output_method(self):
        pass


    def _random_config(self):
        def gen_random(p_info):
            if isinstance(p_info[-1], dict):
                new_vals = {}
                sub_class, sub_params = p_info
                for p_name in sub_params:
                    new_vals[p_name] = gen_random(sub_params[p_name])
                return sub_class(**new_vals)
            else:
                p_type, p_range = p_info
                if p_type == int:
                    return self._rng.randint(p_range[0], p_range[1])
                elif p_type == float:
                    return self._rng.uniform(p_range[0], p_range[1])

        config = {}
        for p_name, p_info in self.params_range.items():
            config[p_name] = gen_random(p_info)

        return config

    def _create_simplex(self, model) -> typing.List:
        # The simplex is divided in:
        # * 0: the best model
        # * 1: the 'good' model
        # * 2: the worst model
        simplex = [None] * 3

        simplex[0] = ModelWrapper(
            self.model.clone(self._random_config()), self.metric.clone()
        )
        simplex[2] = ModelWrapper(
            self.model.clone(self._random_config()), self.metric.clone()
        )

        if self.start == self._START_RANDOM:
            # The intermediate 'good' model is defined randomly
            simplex[1] = ModelWrapper(
                self.model.clone(self._random_config()), self.metric.clone()
            )
        elif self.start == self._START_WARM:
            # The intermediate 'good' model is defined randomly
            simplex[1] = ModelWrapper(copy.deepcopy(model), self.metric.clone())

        return simplex

    def _sort_simplex(self):
        """Ensure the simplex models are ordered by predictive performance."""
        if self.metric.bigger_is_better:
            self._simplex.sort(key=lambda mw: mw.metric.get(), reverse=True)
        else:
            self._simplex.sort(key=lambda mw: mw.metric.get())

    def _nelder_mead_expansion(self) -> typing.Dict:
        """Create expanded models given the simplex models."""

        def apply_operator(param1, param2, func, p_info):
            if isinstance(p_info[1], dict):
                sub_class, sub_params1 = param1
                _, sub_params2 = param2
                sub_info = p_info[1]
                new_params = {}
                for sp_name in sub_params1:
                    sp_info = sub_info[sp_name]
                    # Recursive call to deal with nested hiperparameters
                    new_params[sp_name] = apply_operator(
                        sub_params1[sp_name], sub_params2[sp_name], func, sp_info
                    )
                return sub_class(**new_params)
            else:
                p_type, p_range = p_info
                new_val = func(param1, param2)

                # Range sanity checks
                if new_val < p_range[0]:
                    new_val = p_range[0]
                if new_val > p_range[1]:
                    new_val = p_range[1]

                new_val = round(new_val, 0) if p_type == int else new_val

                return new_val

        def gen_new(m1, m2, func):
            new_config = {}
            m1_params = m1.model._get_params()
            m2_params = m2.model._get_params()

            for p_name, p_info in self.params_range.items():
                new_config[p_name] = apply_operator(
                    m1_params[p_name], m2_params[p_name], func, p_info
                )

            # Modify the current best contender with the new hyperparameter values
            new = ModelWrapper(
                copy.deepcopy(self._simplex[0].model), self.metric.clone()
            )

            return new

        expanded = {}
        # Midpoint between 'best' and 'good'
        expanded["midpoint"] = gen_new(
            self._simplex[0], self._simplex[1], lambda h1, h2: (h1 + h2) / 2
        )

        # Reflection of 'midpoint' towards 'worst'
        expanded["reflection"] = gen_new(
            expanded["midpoint"], self._simplex[2], lambda h1, h2: 2 * h1 - h2
        )
        # Expand the 'reflection' point
        expanded["expansion"] = gen_new(
            expanded["reflection"], expanded["midpoint"], lambda h1, h2: 2 * h1 - h2
        )
        # Shrink 'best' and 'worst'
        expanded["shrink"] = gen_new(
            self._simplex[0], self._simplex[2], lambda h1, h2: (h1 + h2) / 2
        )
        # Contraction of 'midpoint' and 'worst'
        expanded["contraction"] = gen_new(
            expanded["midpoint"], self._simplex[2], lambda h1, h2: (h1 + h2) / 2
        )

        return expanded

    def _nelder_mead_operators(self):
        b = self._simplex[0].metric
        g = self._simplex[1].metric
        w = self._simplex[2].metric
        r = self._expanded["reflection"].metric

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
                    s = self._expanded["shrink"].metric
                    if s.is_better_than(w):
                        self._simplex[2] = self._expanded["shrink"]
                    m = self._expanded["midpoint"].metric
                    if m.is_better_than(g):
                        self._simplex[1] = self._expanded["midpoint"]

        self._sort_simplex()

    @property
    def _models_converged(self) -> bool:
        # Normalize params to ensure they contribute equally to the stopping criterion
        def normalize_flattened_hyperspace(scaled, orig, info, prefix=""):
            for p_name, p_info in info.items():
                prefix_ = prefix + p_name
                if isinstance(p_info[-1], dict):
                    sub_orig = orig[p_name][1]
                    sub_info = p_info[1]
                    prefix_ += "__"
                    normalize_flattened_hyperspace(scaled, sub_orig, sub_info, prefix_)
                else:
                    _, p_range = p_info
                    interval = p_range[1] - p_range[0]
                    scaled[prefix_] = (orig[p_name] - p_range[0]) / interval

        # 1. Simplex in sphere
        scaled_params_b = {}
        scaled_params_g = {}
        scaled_params_w = {}
        normalize_flattened_hyperspace(
            scaled_params_b, self._simplex[0].model._get_params(), self.params_range
        )
        normalize_flattened_hyperspace(
            scaled_params_g, self._simplex[1].model._get_params(), self.params_range
        )
        normalize_flattened_hyperspace(
            scaled_params_w, self._simplex[2].model._get_params(), self.params_range
        )

        max_dist = max(
            [
                utils.math.minkowski_distance(scaled_params_b, scaled_params_g, p=2),
                utils.math.minkowski_distance(scaled_params_b, scaled_params_w, p=2),
                utils.math.minkowski_distance(scaled_params_g, scaled_params_w, p=2),
            ]
        )

        ndim = len(self.params_range)
        r_sphere = max_dist * math.sqrt((ndim / (2 * (ndim + 1))))

        if r_sphere < self.convergence_sphere:
            return True

        # TODO? 2. Simplex did not change

        return False

    @abc.abstractmethod
    def _drift_input(self, y_true, y_pred) -> typing.Union[int, float]:
        pass

    def _learn_converged(self, x, y):
        y_pred = self._best_model.predict_one(x)

        input = self._drift_input(y, y_pred)
        self.drift_detector.update(input)

        # We need to start the optimization process from scratch
        if self.drift_detector.drift_detected:
            self._n = 0
            self._converged = False
            self._simplex = self._create_simplex(self._best_model)

            # There is no proven best model right now
            self._best_model = None
            return

        self._best_model.learn_one(x, y)

    def _learn_not_converged(self, x, y):
        for wrap in self._simplex:
            y_pred = wrap.model.predict_one(x)
            wrap.metric.update(y, y_pred)
            wrap.model.learn_one(x, y)

        # Keep the simplex ordered
        self._sort_simplex()

        if not self._expanded:
            self._expanded = self._nelder_mead_expansion()

        for wrap in self._expanded.values():
            y_pred = wrap.model.predict_one(x)
            wrap.metric.update(y, y_pred)
            wrap.model.learn_one(x, y)

        if self._n == self.grace_period:
            self._n = 0

            # Update the simplex models using Nelder-Mead heuristics
            self._nelder_mead_operators()

            # Discard expanded models
            self._expanded = None

        if self._models_converged:
            self._converged = True
            self._best_model = self._simplex[0].model

    def learn_one(self, x, y):
        self._n += 1

        if self.converged:
            self._learn_converged(x, y)
        else:
            self._learn_not_converged(x, y)

        return self

    @property
    def best_model(self):
        if not self._converged:
            # Lazy selection of the best model
            self._sort_simplex()
            return self._simplex[0].model

        return self._best_model

    @property
    def converged(self):
        return self._converged


class SSPTClassifier(SSPT, base.Classifier):
    """Single-pass Self Parameter Tuning Regressor.

    Parameters
    ----------
    model
    metric
    params_range
    grace_period
    drift_detector
    start
    convergence_sphere
    seed

    References
    ----------
    [1]: Veloso, B., Gama, J., Malheiro, B., & Vinagre, J. (2021).Hyperparameter self-tuning
    for data streams. Information Fusion, 76, 75-86.
    """

    def __init__(
        self,
        model: base.Classifier,
        metric: metrics.base.ClassificationMetric,
        params_range: typing.Dict[str, typing.Tuple],
        grace_period: int = 500,
        drift_detector: base.DriftDetector = drift.ADWIN(),
        start: str = "warm",
        convergence_sphere: float = 0.0001,
        seed: int = None,
    ):
        super().__init__(
            model,
            metric,
            params_range,
            grace_period,
            drift_detector,
            start,
            convergence_sphere,
            seed,
        )

    def _drift_input(self, y_true, y_pred):
        return 0 if y_true == y_pred else 1

    def predict_proba_one(self, x: dict):
        return self.best_model.predict_proba_one(x)


class SSPTRegressor(SSPT, base.Regressor):
    """Single-pass Self Parameter Tuning Regressor.

    Parameters
    ----------
    model
    metric
    params_range
    grace_period
    drift_detector
    start
    convergence_sphere
    seed

    Examples
    --------
    >>> from river import datasets
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import preprocessing
    >>> from river_extra import model_selection

    >>> dataset = datasets.synth.Friedman(seed=42).take(2000)
    >>> reg = preprocessing.StandardScaler() | model_selection.SSPTRegressor(
        model=linear_model.LinearRegressor(),
        metric=metrics.RMSE(),
        params_range={
            "l2": (float, (0.0, 0.5))
        }
    )
    >>> metric = metrics.RMSE()

    >>> for x, y in dataset:
    ...     y_pred = reg.predict_one(x)
    ...     metric.update(y, y_pred)
    ...     reg.learn_one(x, y)

    >>> metric

    References
    ----------
    [1]: Veloso, B., Gama, J., Malheiro, B., & Vinagre, J. (2021).Hyperparameter self-tuning
    for data streams. Information Fusion, 76, 75-86.
    """

    def __init__(
        self,
        model: base.Regressor,
        metric: metrics.base.RegressionMetric,
        params_range: typing.Dict[str, typing.Tuple],
        grace_period: int = 500,
        drift_detector: base.DriftDetector = drift.ADWIN(),
        start: str = "warm",
        convergence_sphere: float = 0.0001,
        seed: int = None,
    ):
        super().__init__(
            model,
            metric,
            params_range,
            grace_period,
            drift_detector,
            start,
            convergence_sphere,
            seed,
        )

    def _drift_input(self, y_true, y_pred):
        return abs(y_true - y_pred)

    def predict_one(self, x: dict):
        return self.best_model.predict_one(x)
