from abc import abstractmethod, ABCMeta

import numpy as np


class RadiusUpdate(object):
    __metaclass__ = ABCMeta

    def __init__(self,
                 step_size=0.01,
                 tolerance=1e-5,
                 burnout_tolerance=1000):
        self.step_size = step_size
        self.tolerance = tolerance
        self.burnout_tolerance = burnout_tolerance

    @abstractmethod
    def update(self, radius, turn, error, prev_error, burnout=0):
        pass


class LinearRadiusDecrease(RadiusUpdate):
    def update(self, radius, turn, error, prev_error, burnout=0):
        #if error > 0.9 * prev_error:
        return max(self.step_size,
                   self.step_size * 10 / np.sqrt(turn)), burnout
        return radius, burnout


class AdaRadiusHalving(RadiusUpdate):
    def _radius_burnout(self, radius, burnout):
        return burnout >= radius * self.burnout_tolerance

    def update(self, radius, turn, error, prev_error, burnout=0):
        if (error >= prev_error or
           prev_error - error <= error * self.tolerance or
           self._radius_burnout(radius, burnout)):
            return radius * 0.5, 0
        return radius, burnout
