from river import datasets
from river import linear_model
from river import metrics
from river import preprocessing
from river_extra import model_selection

dataset = datasets.synth.Friedman(seed=42).take(50000)

sspt = model_selection.SSPT(
    estimator=preprocessing.AdaptiveStandardScaler() | linear_model.LinearRegression(),
    metric=metrics.RMSE(),
    grace_period=500,
    params_range={
        "AdaptiveStandardScaler": {
            "alpha": (float, (0.1, 0.9))
        },
        "LinearRegression": {
            "l2": (float, (0.0, 0.2)),
            "optimizer": {
                "lr": {"learning_rate": (float, (0.0091, 0.5))}
            },
            "intercept_lr": {"learning_rate": (float, (0.0001, 0.5))}
        }
    },
    start="random",
    drift_input=lambda yt, yp: abs(yt - yp),
    convergence_sphere=0.00001,
    seed=42
)
metric = metrics.RMSE()
first_print = True

for i, (x, y) in enumerate(dataset):
    y_pred = sspt.predict_one(x)
    metric.update(y, y_pred)
    sspt.learn_one(x, y)

    if sspt.converged and first_print:
        print("Converged at:", i)
        first_print = False

print("Total instances:", i + 1)
print(metric)
print("Best params:")
print(repr(sspt.best))