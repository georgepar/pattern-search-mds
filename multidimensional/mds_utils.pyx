import numpy as np
cimport numpy as np
cimport cython
from numpy cimport ndarray as nd_arr

# don't use np.sqrt - the sqrt function from the C standard library is much
# faster
from libc.math cimport sqrt


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef nd_arr[np.float64_t, ndim=1] dist_from_point(nd_arr[np.float64_t, ndim=2] x, nd_arr[np.float64_t, ndim=1] y):
    cdef:
        Py_ssize_t nrow = x.shape[0]
        Py_ssize_t ncol = y.shape[0]
        Py_ssize_t ii = 0, jj = 0

        double s = 0, diff = 0

        nd_arr[np.float64_t, ndim=1] d = np.zeros(nrow, np.double)

    for ii in range(nrow):
        s = 0
        for jj in range(ncol):
            diff = x[ii, jj] - y[jj]
            s += diff * diff
        s = sqrt(s)
        d[ii] = s
    return d


# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef nd_arr[np.float64_t, ndim=1] cdist_from_point(nd_arr[np.float64_t, ndim=2] x, nd_arr[np.float64_t, ndim=1] y):
#     cdef:
#         Py_ssize_t nrow = x.shape[0]
#         Py_ssize_t ncol = y.shape[0]
#         Py_ssize_t ii = 0, jj = 0
#
#         double s = 0, diff = 0
#         nd_arr[np.float64_t, ndim=1] d = np.zeros(nrow, np.double)
#
#     for ii in range(nrow):
#         s = 0
#         for jj in range(ncol):
#             diff = x[ii, jj] - y[jj]
#             s += diff * diff
#         s = sqrt(s)
#         d[ii] = s
#     return d


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double mse(nd_arr[np.float64_t, ndim=1] d_goal, nd_arr[np.float64_t, ndim=1] d):
    cdef:
        Py_ssize_t N = d.shape[0]
        Py_ssize_t ii = 0

        double s = 0, diff = 0

    for ii in range(N):
        diff = d_goal[ii] - d[ii]
        s += diff * diff
    return s


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double mse2(nd_arr[np.float64_t, ndim=2] d_goal, nd_arr[np.float64_t, ndim=2] d):
    cdef:
        Py_ssize_t N = d.shape[0]
        Py_ssize_t ii = 0

        double s = 0, diff = 0

    for ii in range(N):
        for jj in range(ii + 1):
            diff = d_goal[ii, jj] - d[ii, jj]
            s += diff * diff
    return s


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double l1e(nd_arr[np.float64_t, ndim=1] d_goal, nd_arr[np.float64_t, ndim=1] d):
    cdef:
        Py_ssize_t N = d.shape[0]
        Py_ssize_t ii = 0

        double s = 0, diff = 0

    for ii in range(N):
        diff = d_goal[ii] - d[ii]
        diff = diff if diff >= 0 else -diff
        s += diff
    return s


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double l1e2(nd_arr[np.float64_t, ndim=2] d_goal, nd_arr[np.float64_t, ndim=2] d):
    cdef:
        Py_ssize_t N = d.shape[0]
        Py_ssize_t ii = 0

        double s = 0, diff = 0

    for ii in range(N):
        for jj in range(ii + 1):
            diff = d_goal[ii, jj] - d[ii, jj]
            diff = diff if diff >= 0 else -diff
            s += diff
    return s

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef nd_arr[np.float64_t, ndim=2] update_distance_matrix(
        nd_arr[np.float64_t, ndim=2] xs,
        nd_arr[np.float64_t, ndim=2] d_current,
        int ii,
        double optimum_step,
        int optimum_k):
    cdef:
        Py_ssize_t N = d_current.shape[0]
        Py_ssize_t jj = 0
        double d = 0

    for jj in range(N):
        if ii != jj:
            d = sqrt(d_current[ii, jj] ** 2 -
                (xs[ii, optimum_k] - xs[jj, optimum_k]) ** 2 +
                (xs[ii, optimum_k] + optimum_step - xs[jj, optimum_k]) ** 2)
            d_current[ii, jj] = d
            d_current[jj, ii] = d
    d_current[ii, ii] = 0
    return d_current


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef (double, int, double) pertub_error(nd_arr[np.float64_t, ndim=2] xs,
                                  double radius,
                                  nd_arr[np.float64_t, ndim=2] d_current,
                                  nd_arr[np.float64_t, ndim=2] d_goal,
                                  int ii,
                                  double percent=.5):
    cdef:
        Py_ssize_t step_size = xs.shape[1]
        Py_ssize_t x_rows = xs.shape[0]
        Py_ssize_t jj, kk, ll


        double optimum_error = np.Inf
        int optimum_k = 0
        double optimum_step = 0
        double e = 0
        double d_temp = 0

    if step_size >= 50:
        ks = np.random.choice(np.arange(0, 2 * step_size), size=int(2 * percent * step_size), replace=False)
    else:
        ks = np.arange(0, 2 * step_size)

    for jj in ks:
        step = radius if jj < step_size else -radius
        kk = jj % step_size
        e = 0
        for ll in range(x_rows):
            d_temp = d_current[ii, ll]
            if ii != ll:
                d_temp = sqrt(
                    d_current[ii, ll] * d_current[ii, ll] -
                    (xs[ii, kk] - xs[ll, kk]) * (xs[ii, kk] - xs[ll, kk]) +
                    (xs[ii, kk] + step - xs[ll, kk]) * (xs[ii, kk] + step - xs[ll, kk]))
            e += (d_goal[ii, ll] - d_temp) ** 2
        if e < optimum_error:
            optimum_error = e
            optimum_k = kk
            optimum_step = step
    return optimum_error, optimum_k, optimum_step