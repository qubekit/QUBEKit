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

        print('{} ran in: {} seconds.'.format(orig_func.__name__, t2))

        return result
    return wrapper


def timer_logger(orig_func):
    """Logs the runtime of a function in a dated and numbered file.
    Outputs the start time, runtime, and function name and docstring.
    Run number can be changed with -log command."""

    from time import time
    from datetime import datetime
    from functools import wraps
    from os import listdir, path

    @wraps(orig_func)
    def wrapper(*args, **kwargs):

        start_time = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
        t1 = time()

        result = orig_func(*args, **kwargs)

        time_taken = time() - t1

        # Find all files in current directory.
        files = [f for f in listdir('.') if path.isfile(f)]

        for file in files:
            # Find log file.
            if file.startswith('QUBEKit_log'):
                file_name = file

                with open(file_name, 'a+') as log_file:
                    log_file.write(f'{orig_func.__name__} began at {start_time}.\n\nDocstring for {orig_func.__name__}: {orig_func.__doc__}\n\n')
                    log_file.write(f'{orig_func.__name__} finished in {time_taken} seconds.')
                    # Add some separation space between function / method logs.
                    log_file.write('\n\n-------------------------------------------------------\n\n')
        return result
    return wrapper


def for_all_methods(decorator):
    """Applies decorator to all methods of a class (includes sub-classes and init; it is literally all callables).
    This class decorator is applied using '@for_all_methods(timer_func)' for example."""

    def decorate(cls):
        # Examine all class attributes.
        for attr in cls.__dict__:
            # Check if each class attribute is a callable method.
            if callable(getattr(cls, attr)):
                # Set the callables to be decorated.
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate
