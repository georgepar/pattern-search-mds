#!/usr/bin/env python3

import sys
import random
import os
import gc
import math
import copy
import numpy as np
import heapq

number_of_rows_in_spiral = 116  # 1404 points
depth_of_spiral = 12
angle_start = math.pi  # 180 deg
angle_stop = 4 * math.pi  # 720 deg
r_start = 0.1
r_stop = 0.9
angle_hole_start = (360 + 45) * math.pi / 180  # 360+45 deg
angle_hole_stop = (360 + 135) * math.pi / 180  # 360+135 deg
use_noise = False
noise_standard_deviation = 10 ** (-2)
show_plots = 0
closest_neighbors_for_dijkstra = 8  # The number of the closest neighbors that one of the vectors will see as not infinitely
# distanced from itself during geodesic calculation.
# This number includes the vector itself

use_distance_for_closest_neighbors = True  # Finds closest neighbors that satisfy the fact that they are closer
# to a vector than a given distance, and not using a fixed number of vectors

distance_for_closest_neighbors = 0  # The distance that will guarantee that the farthest vector will have <closest_neighbors_for_dijkstra-1> neighbors
epsilon = 10 ** (-9)
eucl_distances = []
geodesic_distances = []
eucl_temp = []
value = 0
neighbors_increasing = []
neigh_temp = []
infinite_distance = 1000000000
dijkstra_distances = []
visited_nodes = []

total_rounds = (angle_stop - angle_start) / (2 * math.pi)
points = []
points_with_hole = []


def check_float_equality(a, b):
    if (abs(a - b) < epsilon):
        return 1
    else:
        return 0


# assigns the geodesic distances. They are equal to the euclidean distances for the closest neighbors,
# but they are infinite (for now) for the rest.
# Also, initialises the heaps that will contain the dijkstra distances. At the beginning, these heaps just contain the geodesic distances.
def init_geodesic_distances(n):
    global geodesic_distances
    global neighbors_increasing
    global eucl_distances
    global dijkstra_distances

    geodesic_distances = copy.deepcopy(eucl_distances)
    for i in range(n):
        dijkstra_distances.append([])
        for j in range(n):
            if (use_distance_for_closest_neighbors == False):
                if (j < closest_neighbors_for_dijkstra):
                    geodesic_distances[i][neighbors_increasing[i][j]] = \
                        eucl_distances[i][neighbors_increasing[i][j]]

                else:
                    geodesic_distances[i][
                        neighbors_increasing[i][j]] = infinite_distance
            else:
                if (eucl_distances[i][neighbors_increasing[i][
                    j]] <= distance_for_closest_neighbors):
                    geodesic_distances[i][neighbors_increasing[i][j]] = \
                        eucl_distances[i][neighbors_increasing[i][j]]
                else:
                    geodesic_distances[i][
                        neighbors_increasing[i][j]] = infinite_distance
    for i in range(n):
        for j in range(n):
            # the heap contains tuples of value (<distance>,<node that the distance is to (source node is always i)>)
            heapq.heappush(dijkstra_distances[i], (
                geodesic_distances[i][neighbors_increasing[i][j]],
                neighbors_increasing[i][j]))


def calc_geodesic(n):
    global visited_nodes
    global geodesic_distances
    global neighbors_increasing
    global eucl_distances
    global dijkstra_distances
    global distance_for_closest_neighbors

    for i in range(n):  # perform dijkstra for each one
        visited_nodes = [0 for j in range(n)]  # no-one is visited
        visited_nodes[i] = 1  # except the one we start dijkstra from
        dist = heapq.heappop(
            dijkstra_distances[i])  # fetch the top of the heap
        while (dist[0] < infinite_distance):
            if (visited_nodes[dist[1]] == 1):
                if len(dijkstra_distances[i]) == 0:
                    break
                dist = heapq.heappop(dijkstra_distances[
                                         i])  # if visited, then fetch the next one
                continue
            visited_nodes[dist[1]] = 1  # we visited this new node
            geodesic_distances[i][dist[1]] = dist[
                0]  # update the correct geodesic distance
            if (use_distance_for_closest_neighbors == False):
                for j in range(min(n,
                                   closest_neighbors_for_dijkstra)):  # for every node that does not have an infinite distance from this new node
                    if (visited_nodes[neighbors_increasing[dist[1]][
                        j]] == 0):  # and of course is not visited yet
                        heapq.heappush(dijkstra_distances[i], (
                            dist[0] + eucl_distances[dist[1]][
                                neighbors_increasing[dist[1]][j]],
                            neighbors_increasing[dist[1]][
                                j]))  # push the new distance into the heap
            else:
                j = 0
                while (j < n and eucl_distances[dist[1]][
                    neighbors_increasing[dist[1]][
                        j]] <= distance_for_closest_neighbors):  # for every node that has at most <max_distance> distance from that node
                    if (visited_nodes[neighbors_increasing[dist[1]][
                        j]] == 0):  # and of course is not visited yet
                        heapq.heappush(dijkstra_distances[i], (
                            dist[0] + eucl_distances[dist[1]][
                                neighbors_increasing[dist[1]][j]],
                            neighbors_increasing[dist[1]][
                                j]))  # push the new distance into the heap
                    j += 1
            if len(dijkstra_distances[i]) > 0:
                dist = heapq.heappop(
                    dijkstra_distances[i])  # get the next top of the heap
    gc.collect()


def check_if_symmetric(dm, n):
    for i in range(n):
        for j in range(n):
            if (check_float_equality(dm[i][j], dm[j][i]) == 0):
                print("Not symmetric!: i=%d,j=%d, [i][j]=%lf, [j][i]=%lf" % (
                    i, j, dm[i][j], dm[j][i]))
                return 0
    return 1


def calculate_max_distance_for_having_some_neghbors(n):
    global neighbors_increasing
    global eucl_distances
    global distance_for_closest_neighbors

    max_distance = -1
    for i in range(n):
        achieved_neighbors = 0
        while (
                achieved_neighbors < n and  achieved_neighbors < closest_neighbors_for_dijkstra):
            j = achieved_neighbors
            if (max_distance < eucl_distances[i][neighbors_increasing[i][j]]):
                max_distance = eucl_distances[i][neighbors_increasing[i][j]]
            achieved_neighbors += 1
    distance_for_closest_neighbors = max_distance


if (len(sys.argv) == 2):
    show_plots = int(sys.argv[1])
if (len(sys.argv) == 3):
    show_plots = int(sys.argv[1])
    use_noise = int(sys.argv[2])

angle = angle_start
angle_step = (angle_stop - angle_start) / number_of_rows_in_spiral
distance_step = (r_stop - r_start) / number_of_rows_in_spiral
distance = r_start
while (angle <= angle_stop):
    for i in range(depth_of_spiral):
        x_coord = -0.9 + (1.8 * i) / (
            depth_of_spiral - 1)  # range is [-0.9,0,9]
        y_coord = distance * math.cos(angle)
        z_coord = distance * math.sin(angle)
        if (use_noise):
            noise = np.random.normal(0, noise_standard_deviation, 3)
            x_coord += noise[0]
            y_coord += noise[1]
            z_coord += noise[2]
        x_coord = round(x_coord, 6)  # 6 decimals
        y_coord = round(y_coord, 6)
        z_coord = round(z_coord, 6)
        points.append([x_coord, y_coord, z_coord])
        if (angle_hole_stop >= angle >= angle_hole_start and
                        round(int(2 * depth_of_spiral / 3)) > i >= round(
                    int(depth_of_spiral / 3))):
            pass
        else:
            points_with_hole.append([x_coord, y_coord, z_coord])

    distance += distance_step
    angle += angle_step

n = len(points)
# EUCLIDEAN DISTANCE CALCULATION (NO HOLES)
eucl_distances = []
filename = "Euclidean_spiral.dat"
file_handler = open(filename, 'w')
for ni1 in range(n):
    eucl_temp = []
    for ni2 in range(n):
        value = 0
        for di in range(3):
            value += (points[ni1][di] - points[ni2][di]) ** 2
        value = math.sqrt(value)
        value = round(value, 7)  # 7 decimals
        eucl_temp.append(value)
        file_handler.write(str(value))
        if (ni2 < n - 1):
            file_handler.write(",")
    file_handler.write("\n")
    eucl_distances.append(eucl_temp)
file_handler.close()

# GEODESIC DISTANCE CALCULATION (NO HOLES)
geodesic_distances = []
neighbors_increasing = []
dijkstra_distances = []
visited_nodes = []
filename = "Geodesic_spiral.dat"
file_handler = open(filename, 'w')

for ni1 in range(n):
    neigh_temp = []
    # from here: https://stackoverflow.com/questions/6422700/how-to-get-indices-of-a-sorted-array-in-python
    neigh_temp = [i[0] for i in
                  sorted(enumerate(eucl_distances[ni1]), key=lambda x: x[1])]
    neighbors_increasing.append(
        neigh_temp)  # the indexes of the closest neighbors in ascending order
if (use_distance_for_closest_neighbors):
    calculate_max_distance_for_having_some_neghbors(n)
init_geodesic_distances(n)
calc_geodesic(n)
check_if_symmetric(geodesic_distances, n)

for ni1 in range(n):
    for ni2 in range(n):
        file_handler.write(str(round(geodesic_distances[ni1][ni2], 7)))
        if (ni2 < n - 1):
            file_handler.write(",")
    file_handler.write("\n")

file_handler.close()

print("Max distance under which neighbors are taken into account (no holes):",
      distance_for_closest_neighbors)
max_geod = max([geodesic_distances[i][j] for i in range(n) for j in range(n)])
print("Max geodesic distance between two points:", max_geod)

n = len(points_with_hole)
# EUCLIDEAN DISTANCE CALCULATION (HOLE)
eucl_distances = []
filename = "Euclidean_spiral_with_hole.dat"
file_handler = open(filename, 'w')
for ni1 in range(n):
    eucl_temp = []
    for ni2 in range(n):
        value = 0
        for di in range(3):
            value += (points_with_hole[ni1][di] - points_with_hole[ni2][
                di]) ** 2
        value = math.sqrt(value)
        value = round(value, 7)  # 7 decimals
        eucl_temp.append(value)
        file_handler.write(str(value))
        if (ni2 < n - 1):
            file_handler.write(",")
    file_handler.write("\n")
    eucl_distances.append(eucl_temp)
file_handler.close()

# GEODESIC DISTANCE CALCULATION (HOLE)
geodesic_distances = []
neighbors_increasing = []
dijkstra_distances = []
visited_nodes = []
filename = "Geodesic_spiral_with_hole.dat"
file_handler = open(filename, 'w')

for ni1 in range(n):
    neigh_temp = []
    # from here: https://stackoverflow.com/questions/6422700/how-to-get-indices-of-a-sorted-array-in-python
    neigh_temp = [i[0] for i in
                  sorted(enumerate(eucl_distances[ni1]), key=lambda x: x[1])]
    neighbors_increasing.append(
        neigh_temp)  # the indexes of the closest neighbors in ascending order
if (use_distance_for_closest_neighbors):
    calculate_max_distance_for_having_some_neghbors(n)
init_geodesic_distances(n)
calc_geodesic(n)
check_if_symmetric(geodesic_distances, n)

for ni1 in range(n):
    for ni2 in range(n):
        file_handler.write(str(round(geodesic_distances[ni1][ni2], 7)))
        if (ni2 < n - 1):
            file_handler.write(",")
    file_handler.write("\n")

file_handler.close()

print("Max distance under which neighbors are taken into account (hole):",
      distance_for_closest_neighbors)
max_geod = max([geodesic_distances[i][j] for i in range(n) for j in range(n)])
print("Max geodesic distance between two points:", max_geod)

filename = "spiral_coords_with_hole.dat"
file_handler = open(filename, 'w')
for i in range(len(points_with_hole)):
    for j in range(3):
        file_handler.write(str(points_with_hole[i][j]))
        if (j < 2):
            file_handler.write(",")
    file_handler.write("\n")
file_handler.close()

print("Length of full spiral:", len(points))
print("Length of spiral with hole:", len(points_with_hole))

filename = "spiral_coords.dat"
file_handler = open(filename, 'w')
for i in range(len(points)):
    for j in range(3):
        file_handler.write(str(points[i][j]))
        if (j < 2):
            file_handler.write(",")
    file_handler.write("\n")
file_handler.close()

if (show_plots):
    # PLOT
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib
    import matplotlib.pyplot as plt

    x_coords = [k[0] for k in points]
    y_coords = [k[1] for k in points]
    z_coords = [k[2] for k in points]

    x_coords_hole = [k[0] for k in points_with_hole]
    y_coords_hole = [k[1] for k in points_with_hole]
    z_coords_hole = [k[2] for k in points_with_hole]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_coords, y_coords, z_coords)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_coords_hole, y_coords_hole, z_coords_hole)
    plt.show()
