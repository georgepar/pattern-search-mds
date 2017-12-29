import errno
import logging
import os
import shutil
import stat

import matplotlib.pyplot as plt
import numpy as np

import common
import config

from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D


logging.basicConfig()
LOG = logging.getLogger("MDS History")
LOG.setLevel(logging.DEBUG)


class HistoryObserver(object):
    def __init__(self, path=str(datetime.now().strftime("%Y%m%d%H%M%S"))):
        self.path = os.path.join(config.HISTORY_DIR, path)
        self.history = {
            'radius': [],
            'error': [],
            'xs_files': [],
            'xs_images': [],
            'animation': None
        }
        if os.path.exists(self.path):
            shutil.rmtree(self.path, ignore_errors=True)
        common.mkdir_p(self.path)

    def epoch(self, turn, radius, error, xs):
        xs_file = os.path.join(self.path, 'epoch_{}'.format(turn))
        np.savetxt(xs_file, xs, delimiter=',')
        self.history['radius'].append(radius)
        self.history['error'].append(error)
        self.history['xs_files'].append(xs_file)

    def plot(self, num_epochs, dimensions, color=None):
        if color is not None:
            np.savetxt(os.path.join(self.path, 'color'), color, delimiter=',')
        for epoch in range(num_epochs):
            print('Plotting epoch {}'.format(epoch))
            if dimensions == 3:
                ax = plt.axes(projection='3d')
            else:
                ax = plt.axes()
            plt.title("Epoch " + str(epoch))
            epoch_file = os.path.join(self.path, 'epoch_{}'.format(epoch))
            vectors = np.loadtxt(epoch_file, delimiter=',')

            xx = vectors[:, 0]
            yy = vectors[:, 1]
            if dimensions == 3:
                zz = vectors[:, 2]
                if color is None:
                    ax.scatter(xx, yy, zz)
                else:
                    ax.scatter(xx, yy, zz, c=color, cmap=plt.cm.Spectral)
            if dimensions == 2:
                if color is None:
                    ax.scatter(xx, yy)
                else:
                    ax.scatter(xx, yy, c=color, cmap=plt.cm.Spectral)
            plt.draw()
            epoch_image = os.path.join(self.path, "epoch_{}.png".format(epoch))
            plt.savefig(epoch_image)
            self.history['xs_images'].append(epoch_image)

            ax.clear()

        self.path.rstrip('/')
        epoch_images = '{}/epoch_%d.png'.format(self.path)
        animation = os.path.join(self.path, 'animation.gif')
        os.system("ffmpeg -framerate 3  -i {} {}"
                  .format(epoch_images, animation))
        self.history['animation'] = animation

    def plot2d(self, num_epochs, color=None):
        self.plot(num_epochs, 2, color=color)

    def plot3d(self, num_epochs, color=None):
        self.plot(num_epochs, 3, color=color)
