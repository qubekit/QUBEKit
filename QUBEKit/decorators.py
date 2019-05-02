#!/usr/bin/env python

from QUBEKit.helpers import pretty_print, unpickle

from datetime import datetime
from functools import wraps
import logging
from time import time
import os


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
    """
    Logs the various timings of a function in a dated and numbered file.
    Writes the start time, function / method qualname and docstring when function / method starts.
    Then outputs the runtime and time when function / method finishes.
    """

    @wraps(orig_func)
    def wrapper(*args, **kwargs):

        start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        t1 = time()

        log_file_path = '../QUBEKit_log.txt'

        with open(log_file_path, 'a+') as log_file:
            log_file.write(f'{orig_func.__qualname__} began at {start_time}.\n\n')
            log_file.write(f'Docstring for {orig_func.__qualname__}:\n     {orig_func.__doc__}\n\n')

            time_taken = time() - t1

            mins, secs = divmod(time_taken, 60)
            hours, mins = divmod(mins, 60)

            secs, remain = str(float(secs)).split('.')

            time_taken = f'{int(hours):02d}h:{int(mins):02d}m:{int(secs):02d}s.{remain[:5]}'
            end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            log_file.write(f'{orig_func.__qualname__} finished in {time_taken} at {end_time}.\n\n')
            # Add some separation space between function / method logs.
            log_file.write(f'{"-" * 50}\n\n')

        return orig_func(*args, **kwargs)
    return wrapper


def for_all_methods(decorator):
    """
    Applies a decorator to all methods of a class (includes sub-classes and init; it is literally all callables).
    This class decorator is applied using '@for_all_methods(timer_func)' for example.
    """

    @wraps(decorator)
    def decorate(cls):
        # Examine all class attributes.
        for attr in cls.__dict__:
            # Check if each class attribute is a callable method.
            if callable(getattr(cls, attr)):
                # Set the callables to be decorated.
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate


def logger_format():
    """
    Creates logging object to be returned. Contains proper formatting and locations for logging exceptions.
    This isn't a decorator itself but is only used by exception_logger_decorator so it makes sense for it to be here.
    """

    logger = logging.getLogger('Exception Logger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler('QUBEKit_log.txt')

    # Format the log message
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def exception_logger(func):
    """
    Decorator which logs exceptions to QUBEKit_log file if one occurs.
    Do not apply this decorator to a function / method unless a log file exists in the working dir.
    On exception, the full stack trace is printed to the log file,
    as well as the Ligand class objects which are taken from the pickle file.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logger_format()

        # Run as normal
        try:
            return func(*args, **kwargs)

        except KeyboardInterrupt:
            raise
        # Any other exception that occurs is logged
        except:
            logger.exception(f'An exception occurred with: {func.__qualname__}')
            print(f'An exception occurred with: {func.__qualname__}. View the log file for details.')

            if 'QUBEKit_log.txt' in os.listdir("."):
                log_file = 'QUBEKit_log.txt'
            else:
                log_file = '../QUBEKit_log.txt'

            with open(log_file, 'r') as log:

                # Run through log file backwards to find proper pickle point
                lines = list(reversed(log.readlines()))

                mol_name, pickle_point = False, False
                for pos, line in enumerate(lines):
                    if 'Analysing:' in line:
                        mol_name = line.split()[1]

                    elif ' stage_wrapper' in line:
                        # The stage_wrapper always wraps the method which is the name of the pickle point.
                        pickle_point = lines[pos - 2].split()[-1]

                if not (mol_name and pickle_point):
                    raise EOFError('Cannot locate molecule name or completion stage in log file.')

                mol = unpickle()[pickle_point]
                pretty_print(mol, to_file=True, finished=False)

            # Re-raises the exception
            raise

    return wrapper
