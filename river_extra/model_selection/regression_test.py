import matplotlib.pyplot as plt
from river import datasets, drift, linear_model, metrics, preprocessing, utils

from river_extra import model_selection

# Dataset
dataset = datasets.synth.FriedmanDrift(
    drift_type="gra", position=(7000, 9000), seed=42
).take(10000)


# Baseline - model and metric
baseline_metric = metrics.RMSE()
baseline_rolling_metric = utils.Rolling(metrics.RMSE(), window_size=100)
baseline_metric_plt = []
baseline_rolling_metric_plt = []
baseline = preprocessing.AdaptiveStandardScaler() | linear_model.LinearRegression()

# SSPT - model and metric
sspt_metric = metrics.RMSE()
sspt_rolling_metric = utils.Rolling(metrics.RMSE(), window_size=100)
sspt_metric_plt = []
sspt_rolling_metric_plt = []
sspt = model_selection.SSPT(
    estimator=preprocessing.AdaptiveStandardScaler() | linear_model.LinearRegression(),
    metric=metrics.RMSE(),
    grace_period=100,
    params_range={
        "AdaptiveStandardScaler": {"alpha": (float, (0.25, 0.35))},
        "LinearRegression": {
            "l2": (float, (0.0, 0.0001)),
            "optimizer": {"lr": {"learning_rate": (float, (0.009, 0.011))}},
            "intercept_lr": {"learning_rate": (float, (0.009, 0.011))},
        },
    },
    drift_input=lambda yt, yp: abs(yt - yp),
    drift_detector=drift.PageHinkley(),
    convergence_sphere=0.000001,
    seed=42,
)

first_print = True

metric = metrics.RMSE()


for i, (x, y) in enumerate(dataset):
    baseline_y_pred = baseline.predict_one(x)
    baseline_metric.update(y, baseline_y_pred)
    baseline_rolling_metric.update(y, baseline_y_pred)
    baseline_metric_plt.append(baseline_metric.get())
    baseline_rolling_metric_plt.append(baseline_rolling_metric.get())
    baseline.learn_one(x, y)
    sspt_y_pred = sspt.predict_one(x)
    sspt_metric.update(y, sspt_y_pred)
    sspt_rolling_metric.update(y, sspt_y_pred)
    sspt_metric_plt.append(sspt_metric.get())
    sspt_rolling_metric_plt.append(sspt_rolling_metric.get())
    sspt.learn_one(x, y)

    if sspt.converged and first_print:
        print("Converged at:", i)
        first_print = False

print("Total instances:", i + 1)
print(repr(baseline))
print("Best params:")
print(repr(sspt.best))
print("SSPT: ", sspt_metric)
print("Baseline: ", baseline_metric)


plt.plot(baseline_metric_plt[:10000], linestyle="dotted")
plt.plot(sspt_metric_plt[:10000])
plt.show()

plt.plot(baseline_rolling_metric_plt[:10000], linestyle="dotted")
plt.plot(sspt_rolling_metric_plt[:10000])
plt.show()
