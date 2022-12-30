from scipy.stats import qmc

sampler = qmc.LatinHypercube(d=4)

sample = sampler.random(n=5)

print(sample)

import numpy as np
import matplotlib.pyplot as plt

from smt.sampling_methods import LHS

xlimits = np.array([[0.0, 100.0], [0.0, 1.0], [20.0, 30.0]])
sampling = LHS(xlimits=xlimits)

num = 50
x = sampling(num)

print(x.shape)
plt.plot(x[:, 0], x[:, 1], "o")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for val in x:
    print(val)
    ax.scatter(val[0], val[1], val[2], marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
