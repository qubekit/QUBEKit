#!/usr/bin/env python


from time import time
from datetime import datetime
from functools import wraps
from os import listdir, path
from logging import getLogger, Formatter, FileHandler, INFO

from QUBEKit.helpers import pretty_print, unpickle


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
        # TODO Test try except add narrow down exception.
        try:
            file_name = [file for file in files if file.startswith('QUBEKit_log')][0]
        except:
            file_name = 'temp_QUBE_log'

        with open(file_name, 'a+') as log_file:
            log_file.write(f'{orig_func.__qualname__} began at {start_time}.\n\n')
            log_file.write(f'Docstring for {orig_func.__qualname__}:\n     {orig_func.__doc__}\n\n')

        result = orig_func(*args, **kwargs)

        time_taken = time() - t1

        mins, secs = divmod(time_taken, 60)
        hours, mins = divmod(mins, 60)

        secs, remain = str(float(secs)).split('.')

        time_taken = f'{int(hours):02d}h:{int(mins):02d}m:{int(secs):02d}s.{remain[:5]}'
        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(file_name, 'a+') as log_file:
            log_file.write(f'{orig_func.__qualname__} finished in {time_taken} at {end_time}.\n\n')
            # Add some separation space between function / method logs.
            log_file.write(f'{"-" * 50}\n\n')

        return result
    return wrapper


def for_all_methods(decorator):
    """Applies a decorator to all methods of a class (includes sub-classes and init; it is literally all callables).
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

    return logger, log_file


def exception_logger_decorator(func):
    """Decorator which logs exceptions to QUBEKit_log file if one occurs.
    Do not apply this decorator to a function / method unless a log file exists in the working dir.
    On exception, the full stack trace is printed to the log file,
    as well as the Ligand class objects which are taken from the pickle file.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger, log_file = exception_logger()

        # Run as normal
        try:
            return func(*args, **kwargs)

        # Any exception that occurs is logged
        except:
            logger.exception(f'An exception occurred with: {func.__qualname__}')
            print(f'An exception occurred with: {func.__qualname__}. View the log file for details.')

            with open(log_file, 'r') as log:

                # Run through log file backwards to quickly find proper pickle point
                lines = list(reversed(log.readlines()))

                for pos, line in enumerate(lines):
                    if ' stage_wrapper' in line:
                        # The stage_wrapper always wraps the method which is the name of the pickle point.
                        pickle_point = lines[pos - 2].split()[-1]

                        # Extract the mol name from the log file name
                        mol_name = log_file.split('_')[2]

                        mol = unpickle(f'.{mol_name}_states')[pickle_point]
                        pretty_print(mol, to_file=True, finished=False)

            # Re-raises the exception
            raise

    return wrapper
