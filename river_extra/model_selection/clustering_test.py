from river import datasets, utils, drift, cluster
from river import linear_model
from river import metrics
from river import preprocessing
from river_extra import model_selection
import matplotlib.pyplot as plt

# Dataset
dataset = datasets.Bananas()


# Baseline - model and metric
baseline_metric = metrics.Silhouette()
baseline_rolling_metric = utils.Rolling(metrics.Silhouette(), window_size=100)
baseline_metric_plt=[]
baseline_rolling_metric_plt=[]
baseline = cluster.CluStream()

# SSPT - model and metric
sspt_metric = metrics.Silhouette()
sspt_rolling_metric = utils.Rolling(metrics.Silhouette(), window_size=100)
sspt_metric_plt=[]
sspt_rolling_metric_plt=[]
sspt = model_selection.SSPT(
    estimator=cluster.CluStream(),
    metric=sspt_rolling_metric,
    grace_period=100,
    params_range={
            "max_micro_clusters": (int, (1, 10)),
            "n_macro_clusters": (int, (1, 10)),
            "halflife": (float, (0.1, 0.5))
    },
    start="random",
    drift_input=lambda yt, yp: abs(yt - yp),
    #drift_detector=drift.PageHinkley(),
    convergence_sphere=0.000001,
    seed=42
)

first_print = True



for i, (x, y) in enumerate(dataset):
    baseline_y_pred = baseline.predict_one(x)
    baseline_metric.update(y, baseline_y_pred, baseline.centers)
    baseline_rolling_metric.update(y, baseline_y_pred, baseline.centers)
    baseline_metric_plt.append(baseline_metric.get())
    baseline_rolling_metric_plt.append(baseline_rolling_metric.get())
    baseline.learn_one(x, y)
    sspt_y_pred = sspt.predict_one(x)
    sspt_metric.update(y, sspt_y_pred)
    sspt_rolling_metric.update(y, sspt_y_pred, sspt.estimator.centers)
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
print("SSPT: ",sspt_metric)
print("Baseline: ",baseline_metric)


plt.plot(baseline_metric_plt[:10000], linestyle = 'dotted')
plt.plot(sspt_metric_plt[:10000])
plt.show()

plt.plot(baseline_rolling_metric_plt[:10000], linestyle = 'dotted')
plt.plot(sspt_rolling_metric_plt[:10000])
plt.show()