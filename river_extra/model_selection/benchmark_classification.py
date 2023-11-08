import matplotlib.pyplot as plt
from river import datasets, drift, metrics, preprocessing, tree, utils
from river.datasets import synth

from river_extra.model_selection.bandit_g import Bandit_ps
from river_extra.model_selection.sspt import SSPT
from river_extra.model_selection.bandit_d import Bandit

# Dataset
dataset = datasets.synth.ConceptDriftStream(
    stream=synth.SEA(seed=42, variant=0),
    drift_stream=synth.SEA(seed=40, variant=1),
    seed=1,
    position=10000,
    width=2,
).take(20000)

from river import datasets
#dataset = synth.Hyperplane(seed=42, n_features=2).take(20000)
dataset = datasets.CreditCard().take(20000)

# Baseline - model and metric
baseline_metric = metrics.Accuracy()
baseline_rolling_metric = utils.Rolling(metrics.Accuracy(), window_size=1000)
baseline_metric_plt = []
baseline_rolling_metric_plt = []
baseline = preprocessing.StandardScaler()|tree.HoeffdingTreeClassifier()

# SSPT - model and metric
sspt_metric = metrics.Accuracy()
sspt_rolling_metric = utils.Rolling(metrics.Accuracy(), window_size=1000)
sspt_metric_plt = []
sspt_rolling_metric_plt = []
sspt = SSPT(
    estimator=preprocessing.StandardScaler()|tree.HoeffdingTreeClassifier(),
    metric=sspt_rolling_metric,
    grace_period=1000,
    params_range={
        #"AdaptiveStandardScaler": {"alpha": (float, (0.25, 0.35))},
        "HoeffdingTreeClassifier": {
            "tau": (float, (0.01, 0.09)),
            "delta": (float, (0.00000001, 0.000001)),
            "grace_period": (int, (100, 500)),
        },
    },
    drift_detector=drift.ADWIN(),
    drift_input=lambda yt, yp: 0 if yt == yp else 1,
    convergence_sphere=0.000001,
    seed=42,
)

bandit_metric = metrics.Accuracy()
bandit_rolling_metric = utils.Rolling(metrics.Accuracy(), window_size=1000)
bandit_metric_plt = []
bandit_rolling_metric_plt = []
bandit = Bandit_ps(
    estimator=preprocessing.StandardScaler()|tree.HoeffdingTreeClassifier(),
    metric=bandit_rolling_metric,
    grace_period=1000,
    params_range={
        #"AdaptiveStandardScaler": {"alpha": (float, (0.25, 0.35))},
        "HoeffdingTreeClassifier": {
            "tau": (float, (0.01, 0.1)),
            "delta": (float, (0.00001, 0.001)),
            "grace_period": (int, (100, 1000)),
            "fake_hp": (float, (0.01, 0.1)),
        },
    },
    drift_detector=drift.ADWIN(),
    drift_input=lambda yt, yp: 0 if yt == yp else 1,
    convergence_sphere=0.000001,
    nr_estimators=64,
    seed=42,
)

sspt_first_print = True
bandit_first_print = True

for i, (x, y) in enumerate(dataset):
    if i%1000==0:
        print(i)
    baseline_y_pred = baseline.predict_one(x)
    baseline_metric.update(y, baseline_y_pred)
    baseline_rolling_metric.update(y, baseline_y_pred)
    baseline_metric_plt.append(baseline_metric.get())
    baseline_rolling_metric_plt.append(baseline_rolling_metric.get())
    baseline.learn_one(x, y)
    #print('Baseline-------------')
    #print(baseline2.debug_one(x))




    sspt_y_pred = sspt.predict_one(x)
    sspt_metric.update(y, sspt_y_pred)
    sspt_rolling_metric.update(y, sspt_y_pred)
    sspt_metric_plt.append(sspt_metric.get())
    sspt_rolling_metric_plt.append(sspt_rolling_metric.get())
    sspt.learn_one(x, y)
    #print('Bandit-------------')
    #print(bandit.debug_one(x))
    bandit_y_pred = bandit.predict_one(x)
    bandit_metric.update(y, bandit_y_pred)
    bandit_rolling_metric.update(y, bandit_y_pred)
    bandit_metric_plt.append(bandit_metric.get())
    bandit_rolling_metric_plt.append(bandit_rolling_metric.get())
    bandit.learn_one(x, y)
    if sspt.converged and sspt_first_print:
        print("SSPT Converged at:", i)
        sspt_first_print = False
    if bandit.converged and bandit_first_print:
        print("Bandit Converged at:", i)
        bandit_first_print = False
    if sspt.drift_detector.drift_detected:
        print("SSPT Drift Detected at:",i)
    if bandit.drift_detector.drift_detected:
        print("Bandit Drift Detected at:",i)


print("Total instances:", i + 1)
print(repr(baseline))
print("SSPT Best params:")
print(repr(sspt.best))
print("Bandit Best params:")
print(repr(bandit.best))
print("SSPT: ", sspt_metric)
print("Bandits: ", bandit_metric)
print("Baseline: ", baseline_metric)



plt.plot(baseline_metric_plt[:20000], linestyle="dotted")
plt.plot(sspt_metric_plt[:20000])
plt.show()

plt.plot(baseline_rolling_metric_plt[0:40000],'k')
plt.plot(bandit_rolling_metric_plt[0:40000], 'r')
plt.plot(sspt_rolling_metric_plt[0:40000], 'b')
plt.show()

with open('/Users/brunoveloso/Downloads/ensaio11.csv', 'w') as f:

    for i in bandit._pruned_configurations:
        f.write(i+'\n')

