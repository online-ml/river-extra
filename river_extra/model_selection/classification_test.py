from river import datasets, utils, drift, tree
from river import linear_model
from river import metrics
from river import preprocessing
from river.datasets import synth

from river_extra import model_selection
import matplotlib.pyplot as plt

# Dataset
dataset = datasets.synth.ConceptDriftStream(stream=synth.SEA(seed=42, variant=0),
                                    drift_stream=synth.SEA(seed=42, variant=1),
                                 seed=1, position=5000, width=2).take(10000)


# Baseline - model and metric
baseline_metric = metrics.Accuracy()
baseline_rolling_metric = utils.Rolling(metrics.Accuracy(), window_size=100)
baseline_metric_plt=[]
baseline_rolling_metric_plt=[]
baseline = (
    preprocessing.AdaptiveStandardScaler() |
    tree.HoeffdingTreeClassifier()
)

# SSPT - model and metric
sspt_metric = metrics.Accuracy()
sspt_rolling_metric = utils.Rolling(metrics.Accuracy(), window_size=100)
sspt_metric_plt=[]
sspt_rolling_metric_plt=[]
sspt = model_selection.SSPT(
    estimator=preprocessing.AdaptiveStandardScaler() | tree.HoeffdingTreeClassifier(),
    metric=sspt_rolling_metric,
    grace_period=100,
    params_range={
        "AdaptiveStandardScaler": {
            "alpha": (float, (0.25, 0.35))
        },
        "HoeffdingTreeClassifier": {
            "delta": (float, (0.00001, 0.0001)),
            "grace_period": (int, (100, 500))
        }
    },
    drift_input=lambda yt, yp: abs(yt - yp),
    #drift_detector=drift.PageHinkley(),
    convergence_sphere=0.000001,
    seed=42
)

first_print = True



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
print("SSPT: ",sspt_metric)
print("Baseline: ",baseline_metric)


plt.plot(baseline_metric_plt[:10000], linestyle = 'dotted')
plt.plot(sspt_metric_plt[:10000])
plt.show()

plt.plot(baseline_rolling_metric_plt[:10000], linestyle = 'dotted')
plt.plot(sspt_rolling_metric_plt[:10000])
plt.show()