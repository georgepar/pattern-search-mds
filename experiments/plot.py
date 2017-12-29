import os
import sys

import matplotlib.pyplot as plt

# This import is needed to modify the way figure behaves
from mpl_toolkits.mplot3d import Axes3D
#Axes3D

import numpy as np
#----------------------------------------------------------------------
# Locally linear embedding of the swiss roll

from sklearn import datasets

path = sys.argv[1]
last_epoch = sys.argv[2]
X, color = datasets.samples_generator.make_swiss_roll(n_samples=1500, random_state=42)
X_r = np.loadtxt(os.path.join(path, 'epoch_{}'.format(last_epoch)), delimiter=',')
#----------------------------------------------------------------------
# Plot result

fig = plt.figure()

ax = fig.add_subplot(211, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)

ax.set_title("Original data")
ax = fig.add_subplot(212)
ax.scatter(X_r[:, 0], X_r[:, 1], X_r[:, 2], c=color, cmap=plt.cm.Spectral)
plt.axis('tight')
plt.xticks([]), plt.yticks([])
plt.title('Projected data')
plt.show()
