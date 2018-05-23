import functools
import time


def timefunc(func):
    """
    Decorator that measure the time it takes for a function to complete
    Usage:
      @timefunc
      def time_consuming_function(...):
    """
    @functools.wraps(func)
    def timed(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        elapsed = '{0}'.format(te - ts)
        print('{f} took: {t} sec'.format(f=func.__name__, t=elapsed))
        return result
    return timed


def timemethod(func):
    """
    Decorator that measure the time it takes for a function to complete
    Usage:
      @timemethod
      def time_consuming_method(...):
    """
    @functools.wraps(func)
    def timed(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        elapsed = '{0}'.format(te - ts)
        print('{c}.{f} took: {t} sec'.format(
            c=args[0].__class__.__name__, f=func.__name__, t=elapsed))
        return result
    return timed
