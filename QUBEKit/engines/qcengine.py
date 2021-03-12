#!/usr/bin/env python3

import numpy as np
import qcelemental as qcel
import qcengine as qcng

from QUBEKit.engines.base_engine import Engines
from QUBEKit.utils import constants
from QUBEKit.utils.helpers import check_symmetry


class QCEngine(Engines):
    def __init__(self, molecule):

        super().__init__(molecule)

    def generate_qschema(self):
        """
        Using the molecule object, generate a QCEngine schema. This can then
        be fed into the various QCEngine procedures.
        :param input_type: The part of the molecule object that should be used when making the schema
        :return: The qcelemental qschema
        """

        mol_data = f"{self.molecule.charge} {self.molecule.multiplicity}\n"

        for i, coord in enumerate(self.molecule.coordinates):
            mol_data += f"{self.molecule.atoms[i].atomic_symbol} "
            for item in coord:
                mol_data += f"{item} "
            mol_data += "\n"

        return qcel.models.Molecule.from_data(mol_data)

    def call_qcengine(self, engine, driver):
        """
        Using the created schema, run a particular engine, specifying the driver (job type).
        e.g. engine: geo, driver: energies.
        :param engine: The engine to be used psi4 geometric
        :param driver: The calculation type to be done e.g. energy, gradient, hessian, properties
        :return: The required driver information
        """

        mol = self.generate_qschema()
        options = {"memory": self.molecule.memory, "ncores": self.molecule.threads}

        # Call psi4 for energy, gradient, hessian or property calculations
        if engine == "psi4":
            psi4_task = qcel.models.AtomicInput(
                molecule=mol,
                driver=driver,
                model={"method": self.molecule.theory, "basis": self.molecule.basis},
                keywords={"scf_type": "df"},
            )

            ret = qcng.compute(psi4_task, "psi4", local_options=options)

            if driver == "hessian":
                hess_size = 3 * len(self.molecule.atoms)
                conversion = constants.HA_TO_KCAL_P_MOL / (constants.BOHR_TO_ANGS ** 2)
                hessian = (
                    np.reshape(ret.return_result, (hess_size, hess_size)) * conversion
                )
                check_symmetry(hessian)

                return hessian

            else:
                return ret.return_result

        # Call geometric with psi4 to optimise a molecule
        elif engine == "geometric":
            geo_task = {
                "schema_name": "qcschema_optimization_input",
                "schema_version": 1,
                "keywords": {
                    "coordsys": "dlc",
                    "maxiter": self.molecule.iterations,
                    "program": "psi4",
                    "convergence_set": self.molecule.convergence,
                },
                "input_specification": {
                    "schema_name": "qcschema_input",
                    "schema_version": 1,
                    "driver": "gradient",
                    "model": {
                        "method": self.molecule.theory,
                        "basis": self.molecule.basis,
                    },
                    "keywords": {},
                },
                "initial_molecule": mol,
            }
            return qcng.compute_procedure(
                geo_task, "geometric", return_dict=True, local_options=options
            )

        elif engine == "torchani":
            ani_task = qcel.models.AtomicInput(
                molecule=mol,
                driver=driver,
                model={"method": "ANI1x", "basis": None},
                keywords={"scf_type": "df"},
            )

            return qcng.compute(
                ani_task, "torchani", local_options=options
            ).return_result

        else:
            raise KeyError(
                'Invalid engine type provided. Please use "geometric", "psi4" or "torchani".'
            )
