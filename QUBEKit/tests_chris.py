#! /usr/bin/env python

from QUBEKit.helpers import get_mol_data_from_csv, generate_config_csv, pretty_progress, pretty_print, Configure

from functools import partial


defaults_dict = {'charge': 0, 'multiplicity': 1, 'config': 'default_config'}

qm, fitting, descriptions = Configure.load_config(defaults_dict['config'])
config_dict = [defaults_dict, qm, fitting, descriptions]


def dec_factory(file_name):
    def logger_dec(func):
        def wrapper(*args, **kwargs):
            with open(file_name, 'a+') as log_file:
                log_file.write('working?!')

            return func(*args, **kwargs)
        return wrapper
    return logger_dec


def proper_logger_dec(func, file_name):

    def wrapper(*args, **kwargs):

        with open(file_name, 'a+') as log_file:
            log_file.write('hello!')

        return func(*args, **kwargs)

    return wrapper


fixed_logger_dec = partial(proper_logger_dec, file_name='test2.txt')


@dec_factory('test567.txt')
def some_func(a, b):

    return a + b


class Engine:

    def __init__(self, molecule):

        self.molecule = molecule
        self.name = molecule.name
        self.log_file = molecule.log_file

    def print_name_to_terminal(self):
        print(self.name)


if __name__ == '__main__':

    some_func(2, 3)
