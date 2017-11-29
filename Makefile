CC= gcc
CYTHON = /usr/local/bin/cython

mds_utils:	mds/mds_utils.pyx
		$(CYTHON) mds/mds_utils.pyx
		$(CC) -Wno-cpp -shared -fno-strict-aliasing -fopenmp -ffast-math -O3 -Wall -fPIC -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include/ -o mds/mds_utils.so mds/mds_utils.c

