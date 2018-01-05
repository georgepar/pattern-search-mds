#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy
import os


os.environ['CFLAGS'] = '-Wno-cpp -shared -fno-strict-aliasing -fopenmp -ffast-math -O3 -Wall -fPIC'
setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("mds_utils",
                             sources=["mds_utils.pyx", "parallel_utils.c"],
                             include_dirs=[numpy.get_include(), '/usr/include/python2.7'])],
)
