#!/usr/bin/env python3

from QUBEKit.utils.display import pretty_print
from QUBEKit.utils.helpers import unpickle, COLOURS

from functools import wraps
import logging
import os
from typing import Callable, Optional


def logger_format(log_file):
    """
    Creates logging object to be returned.
    Contains proper formatting and locations for logging exceptions.
    This isn't a decorator itself but is only used by exception_logger so it makes sense for it to be here.
    """

    logger = logging.getLogger('Exception Logger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)

    # Format the log message
    fmt = '\n\n%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def exception_logger(func):
    """
    Decorator which logs exceptions to QUBEKit_log.txt file if one occurs.
    Do not apply this decorator to a function / method unless a log file will exist in the working dir;
    doing so will just raise the exception as normal.

    On any Exception, the Ligand class objects which are taken from the pickle file are printed to the log file,
    then the full stack trace is printed to the log file as well.

    Currently, only Execute.run is decorated like this, as it will always have a log file.
    Decorating other functions this way is possible and won't break anything, but it is pointless.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        # Any BaseException is logged; KeyboardInterrupt and SystemExit must still be raised (see below)
        except BaseException as exc:

            home = getattr(args[0].molecule, 'home', None)
            state = getattr(args[0].molecule, 'state', None)

            if home is None or state is None:
                raise

            mol = unpickle()[state]
            if getattr(args[0].molecule, 'verbose'):
                pretty_print(mol, to_file=True, finished=False)

            log_file = os.path.join(home, 'QUBEKit_log.txt')
            logger = logger_format(log_file)

            logger.exception(f'\nAn exception occurred with: {func.__qualname__}\n')
            print(f'{COLOURS.red}\n\nAn exception occurred with: {func.__qualname__}{COLOURS.end}\n'
                  f'Exception: {exc}\nView the log file for details.')

            if isinstance(exc, SystemExit) or isinstance(exc, KeyboardInterrupt):
                raise

            # Re-raises the exception if it's not a bulk run.
            # Even if the exception is not raised, it is still logged.
            if len(args) >= 1 and hasattr(args[0], 'molecule'):
                if getattr(args[0].molecule, 'bulk_run', None) in [False, None]:
                    raise

    return wrapper


def requires_package(package_name: str, conda_channel: Optional[str] = "conda-forge"):
    """
    This wraps a function to check if the required dependency is available before running.
    """
    from QUBEKit.utils.exceptions import DependencyError

    def inner_decorator(function: Callable):
        @wraps(function)
        def wrapper(*args, **kwargs):
            import importlib

            try:
                importlib.import_module(package_name)
            except (ImportError, ModuleNotFoundError):
                if conda_channel is None:
                    install_method = f"pip install {package_name}."
                else:
                    install_method = f"conda install {package_name} -c {conda_channel}."
                raise DependencyError(f"Missing dependecy for {package_name}. Try installing it with {install_method}")
            except Exception as e:
                raise e

            return function(*args, **kwargs)

        return wrapper

    return inner_decorator
