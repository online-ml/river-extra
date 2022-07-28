import collections
import copy
import functools
import math
import random
import typing

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
    start
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
        start: str = "warm",
        convergence_sphere: float = 0.001,
        seed: int = None,
            #
    ):
        super().__init__()
        self.estimator = estimator
        self.metric = metric
        self.params_range = params_range
        self.drift_input = drift_input

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

        self._best_estimator = None
        self._simplex = self._create_simplex(estimator)

        # Models expanded from the simplex
        self._expanded: typing.Optional[typing.Dict] = None

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

    def _random_config(self):
        def gen_random(p_data, e_data):
            # Sub-component needs to be instantiated
            if isinstance(e_data, tuple):
                sub_class, sub_data = e_data
                sub_config = {}

                for sub_param, sub_info in p_data.items():
                    sub_config[sub_param] = gen_random(sub_info, sub_data[sub_param])
                return sub_class(**sub_config)

            # We reached the numeric parameters
            if isinstance(p_data, tuple):
                p_type, p_range = p_data
                if p_type == int:
                    return self._rng.randint(p_range[0], p_range[1])
                elif p_type == float:
                    return self._rng.uniform(p_range[0], p_range[1])

            # The sub-parameters need to be expanded
            config = {}
            for p_name, p_info in p_data.items():
                e_info = e_data[p_name]
                sub_config = {}
                for sub_name, sub_info in p_info.items():
                    sub_config[sub_name] = gen_random(sub_info, e_info[sub_name])
                config[p_name] = sub_config
            return config

        return gen_random(self.params_range, self.estimator._get_params())

    def _create_simplex(self, model) -> typing.List:
        # The simplex is divided in:
        # * 0: the best model
        # * 1: the 'good' model
        # * 2: the worst model
        simplex = [None] * 3

        simplex[0] = ModelWrapper(
            self.estimator.clone(self._random_config()), self.metric.clone()
        )
        simplex[2] = ModelWrapper(
            self.estimator.clone(self._random_config()), self.metric.clone()
        )

        if self.start == self._START_RANDOM:
            # The intermediate 'good' model is defined randomly
            simplex[1] = ModelWrapper(
                self.estimator.clone(self._random_config()), self.metric.clone()
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

    def _gen_new_estimator(self, e1, e2, func):
        """Generate new configuration given two estimators and a combination function."""

        def apply_operator(param1, param2, p_info, func):
            if isinstance(param1, tuple):
                sub_class, sub_params1 = param1
                _, sub_params2 = param2

                sub_config = {}
                for sub_param, sub_info in p_info.items():
                    sub_config[sub_param] = apply_operator(
                        sub_params1[sub_param], sub_params2[sub_param], sub_info, func
                    )
                return sub_class(**sub_config)
            if isinstance(p_info, tuple):
                p_type, p_range = p_info
                new_val = func(param1, param2)

                # Range sanity checks
                if new_val < p_range[0]:
                    new_val = p_range[0]
                if new_val > p_range[1]:
                    new_val = p_range[1]

                new_val = round(new_val, 0) if p_type == int else new_val
                return new_val

            # The sub-parameters need to be expanded
            config = {}
            for p_name, inner_p_info in p_info.items():
                sub_param1 = param1[p_name]
                sub_param2 = param2[p_name]

                sub_config = {}
                for sub_name, sub_info in inner_p_info.items():
                    sub_config[sub_name] = apply_operator(
                        sub_param1[sub_name], sub_param2[sub_name], sub_info, func
                    )
                config[p_name] = sub_config
            return config

        e1_params = e1.estimator._get_params()
        e2_params = e2.estimator._get_params()

        new_config = apply_operator(e1_params, e2_params, self.params_range, func)
        # Modify the current best contender with the new hyperparameter values
        new = ModelWrapper(
            copy.deepcopy(self._simplex[0].estimator), self.metric.clone()
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
        expanded["contraction"] = self._gen_new_estimator(
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
            if isinstance(orig, tuple):
                _, sub_orig = orig
                for sub_param, sub_info in info.items():
                    prefix_ = prefix + "__" + sub_param
                    normalize_flattened_hyperspace(
                        scaled, sub_orig[sub_param], sub_info, prefix_
                    )
                return

            if isinstance(info, tuple):
                _, p_range = info
                interval = p_range[1] - p_range[0]
                scaled[prefix] = (orig - p_range[0]) / interval
                return

            for p_name, p_info in info.items():
                sub_orig = orig[p_name]
                prefix_ = prefix + "__" + p_name if len(prefix) > 0 else p_name
                normalize_flattened_hyperspace(scaled, sub_orig, p_info, prefix_)

        # 1. Simplex in sphere
        scaled_params_b = {}
        scaled_params_g = {}
        scaled_params_w = {}
        normalize_flattened_hyperspace(
            scaled_params_b, self._simplex[0].estimator._get_params(), self.params_range
        )
        normalize_flattened_hyperspace(
            scaled_params_g, self._simplex[1].estimator._get_params(), self.params_range
        )
        normalize_flattened_hyperspace(
            scaled_params_w, self._simplex[2].estimator._get_params(), self.params_range
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
