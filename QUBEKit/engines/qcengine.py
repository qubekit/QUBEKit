#!/usr/bin/env python3

from QUBEKit.engines.base_engine import Engines

import qcelemental as qcel
import qcengine as qcng



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
            mol_data += f'{self.molecule.atoms[i].atomic_symbol} '
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
        options = {'memory': self.molecule.memory,
                   'ncores': self.molecule.threads}

        # Call psi4 for energy, gradient, hessian or property calculations
        if engine == 'psi4':
            psi4_task = qcel.models.AtomicInput(
                molecule=mol,
                driver=driver,
                model={'method': self.molecule.theory, 'basis': self.molecule.basis},
                keywords={'scf_type': 'df', "scf_properties": ["wiberg_lowdin_indices"]},
            )

            return qcng.compute(psi4_task, 'psi4', local_options=options)

        # Call geometric with psi4 to optimise a molecule
        elif engine == 'geometric':
            geo_task = {
                'schema_name': 'qcschema_optimization_input',
                'schema_version': 1,
                'keywords': {
                    'coordsys': 'dlc',
                    'maxiter': self.molecule.iterations,
                    'program': 'psi4',
                    'convergence_set': self.molecule.convergence,
                    "reset": True,
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
            return qcng.compute_procedure(geo_task, 'geometric', return_dict=True, local_options=options)

        elif engine == 'torchani':
            ani_task = qcel.models.AtomicInput(
                molecule=mol,
                driver=driver,
                model={'method': self.molecule.theory, 'basis': None},
            )

            return qcng.compute(ani_task, 'torchani', local_options=options)

        else:
            raise KeyError('Invalid engine type provided. Please use "geometric", "psi4" or "torchani".')
