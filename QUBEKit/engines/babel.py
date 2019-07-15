#!/usr/bin/env python3

from QUBEKit.utils.decorators import for_all_methods, timer_logger

import subprocess as sp


@for_all_methods(timer_logger)
class Babel:
    """Class to handel babel functions that convert between standard file types
    acts as a thin wrapper around CLI for babel as python bindings require compiling from source."""

    @staticmethod
    def convert(input_file, output_file):
        """
        Convert the given input file type to the required output.
        :param input_file: Input file name, file type is found by splitting the name by .
        :param output_file: Output file name, file type is found by splitting the name by .
        :return: None
        """

        input_type = str(input_file).split(".")[-1]
        output_type = str(output_file).split(".")[-1]

        with open('babel.txt', 'w+') as log:
            sp.run(f'babel -i{input_type} {input_file} -o{output_type} {output_file}',
                   shell=True, stderr=log, stdout=log)
