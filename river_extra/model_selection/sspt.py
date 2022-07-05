import abc
import collections
import typing

from river import base, drift, metrics

ModelWrapper = collections.namedtuple("ModelWrapper", "model metric")


# TODO: change class inheritance
class SSPT(base.Estimator):
    def __init__(
        self,
        model,
        metric: metrics.base.Metric,
        params: typing.Dict[str, typing.Tuple],
        grace_period: int = 500,
        drift_detector: base.DriftDetector = drift.ADWIN(),
    ):
        self.model = model
        self.metric = metric
        self.params = params

        self.grace_period = grace_period
        self.drift_detector = drift_detector

        self._best_model = None

        # [best, good, worst]
        self._simplex = self._create_simplex(model)

        # Models expanded from the simplex
        self._expanded: typing.Optional[typing.List] = None

        self._n = 0
        self._converged = False

    def _create_simplex(self, model) -> typing.List:
        # TODO use namedtuple ModelWrapper to wrap model and performance metric
        pass

    def _expand_simplex(self) -> typing.List:
        # TODO Here happens the model expansion
        pass

    @property
    def _models_converged(self) -> bool:
        # TODO check convergence criteria
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

        if not self._expanded:
            self._expanded = self._expand_simplex()

        for wrap in self._expanded:
            y_pred = wrap.model.predict_one(x)
            wrap.metric.update(y, y_pred)
            wrap.model.learn_one(x, y)

        if self._n == self.grace_period:
            self._n = 0

            # Take the best expanded model and replace the worst contender in the simplex
            if self.metric.bigger_is_better:
                self._expanded.sort(key=lambda w: w.metric.get(), reverse=True)
                self._simplex[-1] = self._expanded[0]
                self._simplex.sort(key=lambda w: w.metric.get(), reverse=True)
            else:
                self._expanded.sort(key=lambda w: w.metric.get())
                self._simplex[-1] = self._expanded[0]
                self._simplex.sort(key=lambda w: w.metric.get())

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

    @property
    def best_model(self):
        if not self._converged:
            return self._simplex[0].model

        return self._best_model

    @property
    def converged(self):
        return self._converged
