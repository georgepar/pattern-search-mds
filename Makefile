CC= gcc
CYTHON = /usr/local/bin/cython

mds_utils:	multidimensional/mds_utils.pyx
		$(CYTHON) multidimensional/mds_utils.pyx
		$(CC) -Wno-cpp -shared -fno-strict-aliasing -fopenmp -ffast-math -O3 -Wall -fPIC -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include/ -o multidimensional/mds_utils.so multidimensional/mds_utils.c

