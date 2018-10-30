#!/usr/bin/env python
"""Various useful decorators."""


def timer_func(orig_func):
    """Prints the runtime of a function when applied as a decorator (@timer_func)."""

    from time import time
    from functools import wraps

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        t1 = time()
        result = orig_func(*args, **kwargs)
        t2 = time() - t1
        print('{} ran in: {} seconds'.format(orig_func.__name__, t2))

        return result
    return wrapper
