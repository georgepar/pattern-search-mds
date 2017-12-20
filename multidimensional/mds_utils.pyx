import numpy as np
cimport numpy as np
cimport cython
from numpy cimport ndarray as nd_arr
from cython.parallel cimport parallel, prange

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
    # for ii in prange(nrow, schedule='static', chunksize=100):
        s = 0
        for jj in range(ncol): #prange(ncol, nogil=False, schedule='static', chunksize=2):
            diff = x[ii, jj] - y[jj]
            s += diff * diff
        s = sqrt(s)
        d[ii] = s
    return d


@cython.boundscheck(False)
@cython.wraparound(False)
cdef nd_arr[np.float64_t, ndim=1] cdist_from_point(nd_arr[np.float64_t, ndim=2] x, nd_arr[np.float64_t, ndim=1] y):
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


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double mse(nd_arr[np.float64_t, ndim=1] d_goal, nd_arr[np.float64_t, ndim=1] d):
    cdef:
        Py_ssize_t N = d.shape[0]
        Py_ssize_t ii = 0

        double s = 0, diff = 0

    for ii in range(N):
    # for ii in prange(N, nogil=False, schedule='static', chunksize=100):
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
    # for ii in prange(N, nogil=False, schedule='static', chunksize=100):
            diff = d_goal[ii, jj] - d[ii, jj]
            s += diff * diff
    return s


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef nd_arr[np.float64_t, ndim=2] update_distance_matrix(
        nd_arr[np.float64_t, ndim=2] xs,
        nd_arr[np.float64_t, ndim=2] d_current,
        int i):
    cdef:
        Py_ssize_t N = d_current.shape[0]
        Py_ssize_t jj = 0
        nd_arr[np.float64_t, ndim=1] norm = cdist_from_point(xs, xs[i, :])

    for jj in range(N):
        d_current[jj, i] = norm[jj]
        d_current[i, jj] = norm[jj]
    return d_current


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef nd_arr[np.float64_t, ndim=2] update_distance_matrix2(
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
cpdef (double, int) pertub_error(nd_arr[np.float64_t, ndim=2] xs,
                                 nd_arr[np.float64_t, ndim=2] xorg,
                                 nd_arr[np.float64_t, ndim=2] steps,
                                 nd_arr[np.float64_t, ndim=2] d_current,
                                 nd_arr[np.float64_t, ndim=2] d_goal,
                                 int ii):
    cdef:
        Py_ssize_t step_size = steps.shape[0]
        Py_ssize_t x_cols = xs.shape[1]
        Py_ssize_t jj, kk

        double optimum_error = np.Inf
        int optimum_k = 0
        double e = np.Inf
        nd_arr[np.float64_t, ndim=2] d

    for kk in range(step_size):
        for jj in range(x_cols):
            xs[ii, jj] = xorg[ii, jj] + steps[kk, jj]
        d = update_distance_matrix(xs, d_current, ii)
        e = mse(d[ii, :], d_goal[ii, :])
        if e < optimum_error:
            optimum_error = e
            optimum_k = kk
    return optimum_error, optimum_k


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef (double, int, double) pertub_error1(nd_arr[np.float64_t, ndim=2] xs,
                                  double radius,
                                  nd_arr[np.float64_t, ndim=2] d_current,
                                  nd_arr[np.float64_t, ndim=2] d_goal,
                                  int ii):
    cdef:
        Py_ssize_t step_size = xs.shape[1]
        Py_ssize_t x_rows = xs.shape[0]
        Py_ssize_t jj, kk, ll

        double optimum_error = np.Inf
        int optimum_k = 0
        double optimum_step = 0
        double e = 0
        double d_temp = 0

    for jj in range(2 * step_size):
        step = radius if jj < step_size else -radius
        kk = jj % step_size
        e = 0
        for ll in range(x_rows):
            d_temp = d_current[ii, ll]
            if ii != ll:
                d_temp = sqrt(d_current[ii, ll] * d_current[ii, ll] -
                                     (xs[ii, kk] - xs[ll, kk]) ** 2 +
                                     (xs[ii, kk] + step - xs[ll, kk]) ** 2)
            e += (d_goal[ii, ll] - d_temp) ** 2
        if e < optimum_error:
            optimum_error = e
            optimum_k = kk
            optimum_step = step
    return optimum_error, optimum_k, optimum_step


"""
@cython.boundscheck(False)
@cython.wraparound(False)
def descent_samples(double [:, :] xs,
                    double [:, :] xorg,
                    double [:, :] d_current,
                    double [:, :] d_goal,
                    double [:, :] steps,
                    double error):

    cdef:
        Py_ssize_t N = d_goal.shape[0]
        Py_ssize_t x_cols = xs.shape[1]

        Py_ssize_t optimum_k = 0
        Py_ssize_t ii = 0, kk = 0, jj = 0

    for ii in range(N):
        error_i = mse(d_current[ii, :], d_goal[ii, :])
 
        optimum_error, optimum_k = pertub_error(xs, xorg, steps, d_current, d_goal, ii)

        for jj in range(x_cols):
            xs[ii, jj] = xorg[ii, jj] + steps[optimum_k, jj]

        d_current = update_distance_matrix(xs, d_current, ii)
        error -= 2 * (error_i - optimum_error)

    return error, d_current, xs

"""
