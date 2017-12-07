#!/usr/bin/env python3

import warnings

import argparse
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

warnings.filterwarnings("ignore", module="matplotlib")


def cmd_args():
    parser = argparse.ArgumentParser(
        description='Plot MDS point history.')
    parser.add_argument(
        '--plot', dest='plot', action='store_true')
    parser.add_argument(
        '--no-plot', dest='plot', action='store_false')
    parser.add_argument(
        '--save', dest='save', action='store_true')
    parser.add_argument(
        '--no-save', dest='save', action='store_false')
    parser.add_argument('--history')

    return parser.parse_args()

args = cmd_args()
history_dir = args.history
save_flag = args.save
plot_flag = args.plot

os.chdir(history_dir)
with open("config") as fd:
    lines = fd.readlines()
    dimensions = int(lines[0].strip().split('=')[1])
    num_epochs = int(lines[1].strip().split('=')[1])

for epoch in range(num_epochs):
    print("Turn:%d" % epoch)
    if dimensions == 3:
        ax = plt.axes(projection='3d')
    else:
        ax = plt.axes()
    plt.title("Epoch " + str(epoch))

    vectors = np.loadtxt('epoch_{}'.format(epoch), delimiter=',')

    xx = vectors[:, 0]
    yy = vectors[:, 1]
    if dimensions == 3:
        zz = vectors[:, 2]
        ax.scatter(xx, yy, zz)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        ax.set_zlim(-1, 1)
    if dimensions == 2:
        ax.plot(xx, yy, 'o')
    plt.draw()

    if plot_flag:
        plt.show(block=False)
        plt.pause(0.5)
    if save_flag:
        plt.savefig("epoch_{}.png".format(epoch))

    ax.clear()

if save_flag:
    os.system("ffmpeg -framerate 3  -i epoch_%d.png animation.gif")
