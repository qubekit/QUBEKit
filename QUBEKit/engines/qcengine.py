#!/usr/bin/env python3

from QUBEKit.engines.base_engine import Engines
from QUBEKit.utils import constants
from QUBEKit.utils.decorators import for_all_methods, timer_logger
from QUBEKit.utils.helpers import check_symmetry

import qcelemental as qcel
import qcengine as qcng

import numpy as np


@for_all_methods(timer_logger)
class QCEngine(Engines):

    def __init__(self, molecule):

        super().__init__(molecule)

    def generate_qschema(self, input_type='input'):
        """
        Using the molecule object, generate a QCEngine schema. This can then
        be fed into the various QCEngine procedures.
        :param input_type: The part of the molecule object that should be used when making the schema
        :return: The qcelemental qschema
        """

        mol_data = f'{self.molecule.charge} {self.molecule.multiplicity}\n'

        for i, coord in enumerate(self.molecule.coords[input_type]):
            mol_data += f'{self.molecule.atoms[i].element} '
            for item in coord:
                mol_data += f'{item} '
            mol_data += '\n'

        return qcel.models.Molecule.from_data(mol_data)

    def call_qcengine(self, engine, driver, input_type):
        """
        Using the created schema, run a particular engine, specifying the driver (job type).
        e.g. engine: geo, driver: energies.
        :param engine: The engine to be used psi4 geometric
        :param driver: The calculation type to be done e.g. energy, gradient, hessian, properties
        :param input_type: The part of the molecule object that should be used when making the schema
        :return: The required driver information
        """

        mol = self.generate_qschema(input_type=input_type)

        # Call psi4 for energy, gradient, hessian or property calculations
        if engine == 'psi4':
            psi4_task = qcel.models.ResultInput(
                molecule=mol,
                driver=driver,
                model={'method': self.molecule.theory, 'basis': self.molecule.basis},
                keywords={'scf_type': 'df'},
            )

            ret = qcng.compute(psi4_task, 'psi4', local_options={'memory': self.molecule.memory,
                                                                 'ncores': self.molecule.threads})

            if driver == 'hessian':
                hess_size = 3 * len(self.molecule.atoms)
                conversion = constants.HA_TO_KCAL_P_MOL / (constants.BOHR_TO_ANGS ** 2)
                hessian = np.reshape(ret.return_result, (hess_size, hess_size)) * conversion
                check_symmetry(hessian)

                return hessian

            else:
                return ret.return_result

        # Call geometric with psi4 to optimise a molecule
        elif engine == 'geometric':
            geo_task = {
                'schema_name': 'qcschema_optimization_input',
                'schema_version': 1,
                'keywords': {
                    'coordsys': 'tric',
                    'maxiter': self.molecule.iterations,
                    'program': 'psi4',
                    'convergence_set': self.molecule.convergence,
                },
                'input_specification': {
                    'schema_name': 'qcschema_input',
                    'schema_version': 1,
                    'driver': 'gradient',
                    'model': {'method': self.molecule.theory, 'basis': self.molecule.basis},
                    'keywords': {},
                },
                'initial_molecule': mol,
            }
            ret = qcng.compute_procedure(
                geo_task, 'geometric', return_dict=True, local_options={'memory': self.molecule.memory,
                                                                        'ncores': self.molecule.threads})
            return ret

        else:
            raise KeyError('Invalid engine type provided. Please use "geo" or "psi4".')
