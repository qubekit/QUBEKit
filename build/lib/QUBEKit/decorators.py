#!/usr/bin/env python


from time import time
from datetime import datetime
from functools import wraps
from os import listdir, path
from logging import getLogger, Formatter, FileHandler, INFO


def timer_func(orig_func):
    """Prints the runtime of a function when applied as a decorator (@timer_func)."""

    @wraps(orig_func)
    def wrapper(*args, **kwargs):

        t1 = time()
        result = orig_func(*args, **kwargs)
        t2 = time() - t1

        print(f'{orig_func.__qualname__} ran in: {t2} seconds.')

        return result
    return wrapper


def timer_logger(orig_func):
    """Logs the various timings of a function in a dated and numbered file.
    Writes the start time, function / method qualname and docstring when function / method starts.
    Then outputs the runtime and time when function / method finishes.
    """

    @wraps(orig_func)
    def wrapper(*args, **kwargs):

        start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        t1 = time()

        # Find all files in current directory; isolate the QUBEKit log file.
        files = [f for f in listdir('.') if path.isfile(f)]
        file_name = [file for file in files if file.startswith('QUBEKit_log')][0]

        with open(file_name, 'a+') as log_file:
            log_file.write(f'{orig_func.__qualname__} began at {start_time}.\n\n')
            log_file.write(f'Docstring for {orig_func.__qualname__}: {orig_func.__doc__}\n\n')

        result = orig_func(*args, **kwargs)

        # TODO Better formatting of time taken? Currently displays in seconds which is a bit messy for longer times.
        time_taken = time() - t1
        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(file_name, 'a+') as log_file:
            log_file.write(f'{orig_func.__qualname__} finished in {time_taken} seconds at {end_time}.\n\n')
            # Add some separation space between function / method logs.
            log_file.write(f'{"-" * 50}\n\n')

        return result
    return wrapper


def for_all_methods(decorator):
    """Applies a decorator to all methods of a class (includes sub-classes and init; it is literally all callables).
    This class decorator is applied using '@for_all_methods(timer_func)' for example.
    """

    def decorate(cls):
        # Examine all class attributes.
        for attr in cls.__dict__:
            # Check if each class attribute is a callable method.
            if callable(getattr(cls, attr)):
                # Set the callables to be decorated.
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate


def exception_logger():
    """Creates logging object to be returned. Contains proper formatting and locations for logging exceptions.
    This isn't a decorator itself but is only used by exception_logger_decorator so it makes sense for it to be here.
    """

    logger = getLogger('Exception Logger')
    logger.setLevel(INFO)

    # Find the log file and set it to be handled.
    files = [file for file in listdir('.') if path.isfile(file)]
    log_file = [file for file in files if file.startswith('QUBEKit_log')][0]
    file_handler = FileHandler(log_file)

    # Format the log message
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = Formatter(fmt)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def exception_logger_decorator(func):
    """Decorator which logs exceptions to QUBEKit_log file if one occurs.
    Do not apply this decorator to a function / method unless a log file has been produced.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = exception_logger()

        # Run as normal
        try:
            return func(*args, **kwargs)

        # Any exception that occurs is logged
        except:
            logger.exception(f'An exception occurred with: {func.__qualname__}')
            print(f'An exception occurred with: {func.__qualname__}. View the log file for details.')

            # TODO Print the ligand class objects to the log file as well. Before or after the exception statement?

            # Re-raises the exception
            # TODO Do we want the exception to be re-raised? Maybe just continue onto next ligand.

            # TODO Add better exception handling for certain issues. e.g.
            #       No such file or directory: 'opt.xyz' should probably check that the charge/multiplicity is correct
            raise

    return wrapper
