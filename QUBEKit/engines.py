#!/usr/bin/env python

# TODO Expand the functional_dict for PSI4 and Gaussian classes to "most" functionals.
# TODO Add better error handling for missing info. (Done for file extraction.)
#       Maybe add path checking for Chargemol?
# TODO use QCEngine to run PSI4, geometric and torsion drive QM commands.

from QUBEKit.helpers import get_overage, check_symmetry, append_to_log
from QUBEKit.decorators import for_all_methods, timer_logger

from subprocess import run as sub_run

from numpy import array, zeros, reshape
from numpy import append as np_append
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rdkit.Chem import AllChem, MolFromPDBFile, Descriptors, MolToSmiles, MolToSmarts, MolToMolFile
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule, UFFOptimizeMolecule

import qcengine as qcng
import qcelemental as qcel


class Engines:
    """
    Engines superclass containing core information that all other engines (PSI4, Gaussian etc) will have.
    Provides atoms' coordinates with name tags for each atom and entire molecule.
    Also gives all configs from the appropriate config file.
    """

    def __init__(self, molecule, config_dict):

        self.molecule = molecule
        self.charge = config_dict[0]['charge']
        self.multiplicity = config_dict[0]['multiplicity']
        self.qm, self.fitting, self.descriptions = config_dict[1:]

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__!r})'


@for_all_methods(timer_logger)
class PSI4(Engines):
    """
    Writes and executes input files for psi4.
    Also used to extract Hessian matrices; optimised structures; frequencies; etc.
    """

    def __init__(self, molecule, config_dict):

        super().__init__(molecule, config_dict)

        self.functional_dict = {'PBEPBE': 'PBE'}
        if self.functional_dict.get(self.qm['theory'], None) is not None:
            self.qm['theory'] = self.functional_dict[self.qm['theory']]

    def generate_input(self, input_type='input', optimise=False, hessian=False, density=False, energy=False,
                       fchk=False, run=True):
        """Converts to psi4 input format to be run in psi4 without using geometric"""

        molecule = self.molecule.molecule[input_type]

        setters = ''
        tasks = ''

        # input.dat is the PSI4 input file.
        with open('input.dat', 'w+') as input_file:
            # opening tag is always writen
            input_file.write(f"memory {self.qm['memory']} GB\n\nmolecule {self.molecule.name} {{\n{self.charge} {self.multiplicity} \n")
            # molecule is always printed
            for atom in molecule:
                input_file.write(f' {atom[0]}    {float(atom[1]): .10f}  {float(atom[2]): .10f}  {float(atom[3]): .10f} \n')
            input_file.write(f" units angstrom\n no_reorient\n}}\n\nset {{\n basis {self.qm['basis']}\n")

            if energy:
                append_to_log('Writing psi4 energy calculation input')
                tasks += f"\nenergy  = energy('{self.qm['theory']}')"

            if optimise:
                append_to_log('Writing PSI4 optimisation input', 'minor')
                setters += f" g_convergence {self.qm['convergence']}\n GEOM_MAXITER {self.qm['iterations']}\n"
                tasks += f"\noptimize('{self.qm['theory'].lower()}')"

            if hessian:
                append_to_log('Writing PSI4 Hessian matrix calculation input', 'minor')
                setters += ' hessian_write on\n'

                tasks += f"\nenergy, wfn = frequency('{self.qm['theory'].lower()}', return_wfn=True)"

                tasks += '\nwfn.hessian().print_out()\n\n'

            if density:
                append_to_log('Writing PSI4 density calculation input', 'minor')
                setters += " cubeprop_tasks ['density']\n"

                overage = get_overage(self.molecule.name)
                setters += " CUBIC_GRID_OVERAGE [{0}, {0}, {0}]\n".format(overage)
                setters += " CUBIC_GRID_SPACING [0.13, 0.13, 0.13]\n"
                tasks += f"grad, wfn = gradient('{self.qm['theory'].lower()}', return_wfn=True)\ncubeprop(wfn)"

            if fchk:
                append_to_log('Writing PSI4 input file to generate fchk file')
                tasks += f"\ngrad, wfn = gradient('{self.qm['theory'].lower()}', return_wfn=True)"
                tasks += '\nfchk_writer = psi4.core.FCHKWriter(wfn)'
                tasks += f'\nfchk_writer.write("{self.molecule.name}_psi4.fchk")\n'

            # TODO If overage cannot be made to work, delete and just use Gaussian.
            # if self.qm['solvent']:
            #     setters += ' pcm true\n pcm_scf_type total\n'
            #     tasks += '\n\npcm = {'
            #     tasks += '\n units = Angstrom\n Medium {\n  SolverType = IEFPCM\n  Solvent = Chloroform\n }'
            #     tasks += '\n Cavity {\n  RadiiSet = UFF\n  Type = GePol\n  Scaling = False\n  Area = 0.3\n  Mode = Implicit'
            #     tasks += '\n }\n}'

            setters += '}\n'

            if not run:
                setters += f'set_num_threads({self.qm["threads"]})\n'

            input_file.write(setters)
            input_file.write(tasks)

        if run:
            with open('log.txt', 'w+') as log:
                sub_run(f'psi4 input.dat -n {self.qm["threads"]}', shell=True, stdout=log, stderr=log)

    def hessian(self):
        """
        Parses the Hessian from the output.dat file (from psi4) into a numpy array.
        Molecule is a numpy array of size N x N.
        """

        hess_size = 3 * len(self.molecule.molecule['input'])

        # output.dat is the psi4 output file.
        with open('output.dat', 'r') as file:

            lines = file.readlines()

            for count, line in enumerate(lines):
                if '## Hessian' in line or '## New Matrix (Symmetry' in line:
                    # Set the start of the hessian to the row of the first value.
                    hess_start = count + 5
                    break
            else:
                raise EOFError('Cannot locate Hessian matrix in output.dat file.')

            # Check if the hessian continues over onto more lines (i.e. if hess_size is not divisible by 5)
            extra = 0 if hess_size % 5 == 0 else 1

            # hess_length: # of cols * length of each col
            #            + # of cols - 1 * #blank lines per row of hess_vals
            #            + # blank lines per row of hess_vals if the hess_size continues over onto more lines.
            hess_length = (hess_size // 5) * hess_size + (hess_size // 5 - 1) * 3 + extra * (3 + hess_size)

            hess_end = hess_start + hess_length

            hess_vals = []

            for file_line in lines[hess_start:hess_end]:
                # Compile lists of the 5 Hessian floats for each row.
                # Number of floats in last row may be less than 5.
                # Only the actual floats are added, not the separating numbers.
                row_vals = [float(val) for val in file_line.split() if len(val) > 5]
                hess_vals.append(row_vals)

            # Remove blank list entries
            hess_vals = [elem for elem in hess_vals if elem]

            reshaped = []

            # Convert from list of (lists, length 5) to 2d array of size hess_size x hess_size
            for old_row in range(hess_size):
                new_row = []
                for col_block in range(hess_size // 5 + extra):
                    new_row += hess_vals[old_row + col_block * hess_size]

                reshaped.append(new_row)

            hess_matrix = array(reshaped)

            # Cache the unit conversion.
            conversion = 627.509391 / (0.529 ** 2)
            hess_matrix *= conversion

            check_symmetry(hess_matrix)

            return hess_matrix

    def optimised_structure(self):
        """
        Parses the final optimised structure from the output.dat file (from psi4) to a numpy array.
        Also returns the energy of the optimized structure.
        """

        # Run through the file and find all lines containing '==> Geometry', add these lines to a list.
        # Reverse the list
        # from the start of this list, jump down to the first atom and set this as the start point
        # Split the row into 4 columns: centre, x, y, z.
        # Add each row to a matrix.
        # Return the matrix.

        # output.dat is the psi4 output file.
        with open('output.dat', 'r') as file:
            lines = file.readlines()
            # Will contain index of all the lines containing '==> Geometry'.
            geo_pos_list = []
            for count, line in enumerate(lines):
                if "==> Geometry" in line:
                    geo_pos_list.append(count)

                elif "**** Optimization is complete!" in line:
                    opt_pos = count
                    opt_steps = int(line.split()[5])

            if not (opt_pos and opt_steps):
                raise EOFError('According to the output.dat file, optimisation has not completed.')

            # now get the final opt_energy
            opt_energy = float(lines[opt_pos + opt_steps + 7].split()[1])

            # Set the start as the last instance of '==> Geometry'.
            start_of_vals = geo_pos_list[-1] + 9

            opt_struct = []

            for row in range(len(self.molecule.molecule['input'])):

                # Append the first 4 columns of each row, converting to float as necessary.
                struct_row = [lines[start_of_vals + row].split()[0]]
                for indx in range(3):
                    struct_row.append(float(lines[start_of_vals + row].split()[indx + 1]))

                opt_struct.append(struct_row)

        return opt_struct, opt_energy

    @staticmethod
    def get_energy():
        """Get the energy of a single point calculation."""

        # open the psi4 log file
        with open('output.dat', 'r') as log:
            lines = log.readlines()

        # find the total converged energy
        for line in lines:
            if 'Total Energy =' in line:
                energy = float(line.split()[3])
                break
        else:
            raise EOFError('Cannot find energy in output.dat file.')

        return energy

    def all_modes(self):
        """Extract all modes from the psi4 output file."""

        # Find "post-proj  all modes"
        # Jump to first value, ignoring text.
        # Move through data, adding it to a list
        # continue onto next line.
        # Repeat until the following line is known to be empty.

        # output.dat is the psi4 output file.
        with open('output.dat', 'r') as file:
            lines = file.readlines()
            for count, line in enumerate(lines):
                if "post-proj  all modes" in line:
                    start_of_vals = count
                    break
            else:
                raise EOFError('Cannot locate modes in output.dat file.')

            # Barring the first (and sometimes last) line, dat file has 6 values per row.
            end_of_vals = start_of_vals + (3 * len(self.molecule.molecule['input'])) // 6

            structures = lines[start_of_vals][24:].replace("'", "").split()
            structures = structures[6:]

            for row in range(1, end_of_vals - start_of_vals):
                # Remove double strings and weird formatting.
                structures += lines[start_of_vals + row].replace("'", "").replace("]", "").split()

            all_modes = [float(val) for val in structures]

            return array(all_modes)

    def geo_gradient(self, input_type='input', threads=False, run=True):
        """
        Write the psi4 style input file to get the gradient for geometric
        and run geometric optimisation.
        """

        molecule = self.molecule.molecule[input_type]

        with open(f'{self.molecule.name}.psi4in', 'w+') as file:

            file.write(f'memory {self.qm["memory"]} GB\n\nmolecule {self.molecule.name} {{\n {self.charge} {self.multiplicity} \n')
            for atom in molecule:
                file.write(f'  {atom[0]:2}    {float(atom[1]): .10f}  {float(atom[2]): .10f}  {float(atom[3]): .10f}\n')

            file.write(f" units angstrom\n no_reorient\n}}\nset basis {self.qm['basis']}\n")

            if threads:
                file.write(f"set_num_threads({self.qm['threads']})")
            file.write(f"\n\ngradient('{self.qm['theory']}')\n")

        if run:
            with open('log.txt', 'w+') as log:
                sub_run(f'geometric-optimize --psi4 {self.molecule.name}.psi4in {self.molecule.constraints_file} --nt {self.qm["threads"]}',
                        shell=True, stdout=log, stderr=log)


@for_all_methods(timer_logger)
class Chargemol(Engines):

    def __init__(self, molecule, config_file):

        super().__init__(molecule, config_file)

    def generate_input(self, run=True):
        """Given a DDEC version (from the defaults), this function writes the job file for chargemol and executes it."""

        if (self.qm['ddec_version'] != 6) and (self.qm['ddec_version'] != 3):
            append_to_log(message='Invalid or unsupported DDEC version given, running with default version 6.',
                          msg_type='warning')
            self.qm['ddec_version'] = 6

        # Write the charges job file.
        with open('job_control.txt', 'w+') as charge_file:

            charge_file.write(f'<input filename>\n{self.molecule.name}.wfx\n</input filename>')

            charge_file.write('\n\n<net charge>\n0.0\n</net charge>')

            charge_file.write('\n\n<periodicity along A, B and C vectors>\n.false.\n.false.\n.false.')
            charge_file.write('\n</periodicity along A, B and C vectors>')

            charge_file.write(f'\n\n<atomic densities directory complete path>\n{self.descriptions["chargemol"]}/atomic_densities/')
            charge_file.write('\n</atomic densities directory complete path>')

            charge_file.write(f'\n\n<charge type>\nDDEC{self.qm["ddec_version"]}\n</charge type>')

            charge_file.write('\n\n<compute BOs>\n.true.\n</compute BOs>')

        # sub_run(f'psi4 input.dat -n {self.qm["threads"]}', shell=True)
        # sub_run('mv Dt.cube total_density.cube', shell=True)

        if run:
            with open('log.txt', 'w+') as log:
                control_path = 'chargemol_FORTRAN_09_26_2017/compiled_binaries/linux/Chargemol_09_26_2017_linux_serial job_control.txt'
                sub_run(f'{self.descriptions["chargemol"]}/{control_path}', shell=True, stdout=log, stderr=log)


@for_all_methods(timer_logger)
class Gaussian(Engines):
    """
    Writes and executes input files for Gaussian09.
    Also used to extract Hessian matrices; optimised structures; frequencies; etc.
    """

    def __init__(self, molecule, config_dict):

        super().__init__(molecule, config_dict)

        self.functional_dict = {'PBE': 'PBEPBE', 'wB97X-D': 'wB97XD'}
        if self.functional_dict.get(self.qm['theory'], None) is not None:
            self.qm['theory'] = self.functional_dict[self.qm['theory']]

    def generate_input(self, input_type='input', optimise=False, hessian=False, density=False, solvent=False, run=True):
        """Generates the relevant job file for Gaussian, then executes this job file."""

        molecule = self.molecule.molecule[input_type]

        with open(f'gj_{self.molecule.name}', 'w+') as input_file:

            input_file.write(f'%Mem={self.qm["memory"]}GB\n%NProcShared={self.qm["threads"]}\n%Chk=lig\n')

            commands = f'# {self.qm["theory"]}/{self.qm["basis"]} SCF=XQC '

            # Adds the commands in groups. They MUST be in the right order because Gaussian.
            if optimise:
                commands += 'opt '

            if hessian:
                commands += 'freq '

            if self.qm['solvent']:
                commands += 'SCRF=(IPCM,Read) '

            if density:
                commands += 'density=current OUTPUT=WFX '

            commands += f'\n\n{self.molecule.name}\n\n{self.charge} {self.multiplicity}\n'

            input_file.write(commands)

            # Add the atomic coordinates
            for atom in molecule:
                input_file.write(f'{atom[0]} {float(atom[1]): .3f} {float(atom[2]): .3f} {float(atom[3]): .3f}\n')

            if solvent:
                # Adds the epsilon and cavity params
                input_file.write('\n4.0 0.0004')

            if density:
                # Specify the creation of the wavefunction file
                input_file.write(f'\n{self.molecule.name}.wfx')

            # Blank lines because Gaussian.
            input_file.write('\n\n')

        if run:
            with open('log.txt', 'w+') as log:
                sub_run(f'g09 < gj_{self.molecule.name} > gj_{self.molecule.name}.log',
                        shell=True, stdout=log, stderr=log)

    def hessian(self):
        """Extract the Hessian matrix from the Gaussian fchk file."""

        with open('lig.fchk', 'r') as fchk:

            lines = fchk.readlines()
            hessian_list = []

            for count, line in enumerate(lines):
                if line.startswith('Cartesian Force Constants'):
                    start_pos = count + 1
                if line.startswith('Dipole Moment'):
                    end_pos = count

            if not start_pos and end_pos:
                raise EOFError('Cannot locate Hessian matrix in lig.fchk file.')

            for line in lines[start_pos: end_pos]:
                # Extend the list with the converted floats from the file, splitting on spaces and removing '\n' tags.
                hessian_list.extend([float(num) * 0.529 for num in line.strip('\n').split()])

        hess_size = 3 * len(self.molecule.molecule['input'])

        hessian = zeros((hess_size, hess_size))

        # Rewrite Hessian to full, symmetric 3N * 3N matrix rather than list with just the non-repeated values.
        m = 0
        for i in range(hess_size):
            for j in range(i + 1):
                hessian[i, j] = hessian_list[m]
                hessian[j, i] = hessian_list[m]
                m += 1

        check_symmetry(hessian)

        return hessian

    def optimised_structure(self):
        """Extract the optimised structure from the Gaussian log file."""

        with open(f'gj_{self.molecule.name}.log', 'r') as log_file:

            lines = log_file.readlines()

            opt_coords_pos = []
            for pos, line in enumerate(lines):
                if 'Input orientation' in line:
                    opt_coords_pos.append(pos + 5)

            start_pos = opt_coords_pos[-1]

            num_atoms = len(self.molecule.molecule['input'])

            opt_struct = []

            for pos, line in enumerate(lines[start_pos: start_pos + num_atoms]):

                vals = line.split()[-3:]
                vals = [self.molecule.molecule['input'][pos][0]] + [float(i) for i in vals]
                opt_struct.append(vals)

        return opt_struct

    def all_modes(self):
        """Extract the frequencies from the Gaussian log file."""

        with open(f'gj_{self.molecule.name}.log', 'r') as gj_log_file:

            lines = gj_log_file.readlines()
            freqs = []

            # Stores indices of rows which will be used
            freq_positions = []
            for count, line in enumerate(lines):
                if line.startswith(' Frequencies'):
                    freq_positions.append(count)

            for pos in freq_positions:
                freqs.extend(float(num) for num in lines[pos].split()[2:])

        return array(freqs)


@for_all_methods(timer_logger)
class ONETEP(Engines):

    def __init__(self, molecule, config_dict):

        super().__init__(molecule, config_dict)

    def generate_input(self, input_type='input', density=False, solvent=False):
        """ONETEP takes a xyz input file."""

        if density:
            self.molecule.write_xyz(input_type=input_type)

        # should we make a onetep run file? this is quite specific?
        print('Run this file in ONETEP.')

    def calculate_hull(self):
        """
        Generate the smallest convex hull which encloses the molecule.
        Then make a 3d plot of the points and hull.
        """

        coords = array([atom[1:] for atom in self.molecule.molecule['input']])

        hull = ConvexHull(coords)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection=Axes3D.name)

        ax.plot(coords.T[0], coords.T[1], coords.T[2], 'ko')

        for simplex in hull.simplices:
            simplex = np_append(simplex, simplex[0])
            ax.plot(coords[simplex, 0], coords[simplex, 1], coords[simplex, 2], color='lightseagreen')

        plt.show()


class QCEngine(Engines):

    def __init__(self, molecule, config_dict):

        super().__init__(molecule, config_dict)

    def generate_qschema(self, input_type='input'):
        """
        Using the molecule object, generate a QCEngine schema. This can then
        be fed into the various QCEngine procedures.
        """

        mol_data = f'{self.charge} {self.multiplicity}\n'

        for coord in self.molecule.molecule[input_type]:
            for item in coord:
                mol_data += f'{item} '
            mol_data += '\n'

        mol = qcel.models.Molecule.from_data(mol_data)

        return mol

    def call_qcengine(self, engine, driver, input_type):
        """
        Using the created schema, run a particular engine, specifying the driver (job type).
        e.g. engine: geo, driver: energies
        """

        # OLD or NEW? Method for getting qcengine to work. May be removed later
        # task = {
        #     "schema_name": "qcschema_input",
        #     "schema_version": 1,
        #     "molecule": self.generate_qschema(input_type=input_type),
        #     "driver": driver,
        #     "model": {"method": self.qm['theory'], "basis": self.qm['basis']},
        #     "keywords": {"scf_type": "df"},
        #     "return_output": False
        # }

        mol = self.generate_qschema(input_type=input_type)

        # Call psi4 for energy, gradient, hessian or property calculations
        if engine == 'psi4':
            psi4_task = qcel.models.ResultInput(
                molecule=mol,
                driver=driver,
                model={'method': self.qm['theory'], 'basis': self.qm['basis']},
                keywords={'scf_type': 'df'},
            )

            ret = qcng.compute(psi4_task, 'psi4', local_options={'memory': self.qm['memory'], 'ncores': self.qm['threads']})

            if driver == 'hessian':
                hess_size = 3 * len(self.molecule.molecule[input_type])
                hessian = reshape(array(ret.return_result) * 627.509391 / (0.529 ** 2), (hess_size, hess_size))
                check_symmetry(hessian)

                return hessian

            else:
                return ret.return_result

        # Call geometric with psi4 to optimise a molecule
        elif engine == 'geometric':
            geo_task = {
                "schema_name": "qcschema_optimization_input",
                "schema_version": 1,
                "keywords": {
                    "coordsys": "tric",
                    "maxiter": self.qm['iterations'],
                    "program": "psi4"
                },
                "input_specification": {
                    "schema_name": "qcschema_input",
                    "schema_version": 1,
                    "driver": 'gradient',
                    "model": {'method': self.qm['theory'], 'basis': self.qm['basis']},
                    "keywords": {},
                },
                "initial_molecule": mol,
            }

            # return_dict=True seems to be default False in newer versions. Ergo docs are wrong again.
            ret = qcng.compute_procedure(geo_task, 'geometric', return_dict=True)
            return ret['trajectory']

        else:
            raise KeyError('Invalid engine type provided. Please use "geo" or "psi4".')


class RDKit:
    """Class for controlling useful RDKit functions try to keep class static."""

    @staticmethod
    def smiles_to_pdb_mol(smiles_string, name=None):
        """Converts smiles strings to pdb and mol files."""
        # Originally written by venkatakrishnan; rewritten and extended by Chris Ringrose

        if 'H' in smiles_string:
            raise SyntaxError('Smiles string contains hydrogen atoms; try again.')

        m = AllChem.MolFromSmiles(smiles_string)
        if not name:
            name = input('Please enter a name for the molecule:\n>')
        m.SetProp('_Name', name)
        m_h = AllChem.AddHs(m)
        AllChem.EmbedMolecule(m_h, AllChem.ETKDG())
        AllChem.SanitizeMol(m_h)

        print(AllChem.MolToMolBlock(m_h), file=open(f'{name}.mol', 'w+'))
        AllChem.MolToPDBFile(m_h, f'{name}.pdb')

        return f'{name}.pdb'

    @staticmethod
    def mm_optimise(pdb_file, ff='MMF'):
        """
        Perform rough preliminary optimisation to speed up later optimisations
        and extract some extra information about the molecule.
        """

        force_fields = {'MMF': MMFFOptimizeMolecule, 'UFF': UFFOptimizeMolecule}

        mol = MolFromPDBFile(pdb_file, removeHs=False)

        force_fields[ff](mol)

        AllChem.MolToPDBFile(mol, f'{pdb_file[:-4]}_rdkit_optimised.pdb')

        return f'{pdb_file[:-4]}_rdkit_optimised.pdb'

    @staticmethod
    def rdkit_descriptors(pdb_file):
        """
        Use RDKit Descriptors to extract properties and store in Descriptors dictionary
        :param pdb_file: The molecule input file
        :return: descriptors dictionary
        """

        mol = MolFromPDBFile(pdb_file, removeHs=False)
        # Use RDKit Descriptors to extract properties and store in Descriptors dictionary
        descriptors = {'Heavy atoms': Descriptors.HeavyAtomCount(mol),
                       'H-bond donors': Descriptors.NumHDonors(mol),
                       'H-bond acceptors': Descriptors.NumHAcceptors(mol),
                       'Molecular weight': Descriptors.MolWt(mol),
                       'LogP': Descriptors.MolLogP(mol)}

        return descriptors

    @staticmethod
    def get_smiles(pdb_file):
        """
        Use RDKit to load in the pdb file of the molecule and get the smiles code.
        :param pdb_file: The molecule input file
        :return: The smiles string
        """

        mol = MolFromPDBFile(pdb_file, removeHs=False)

        return MolToSmiles(mol, isomericSmiles=True, allHsExplicit=True)

    @staticmethod
    def get_smarts(pdb_file):
        """
        Use RDKit to get the smarts string of the molecule.
        :param pdb_file: The molecule input file
        :return: The smarts string
        """

        mol = MolFromPDBFile(pdb_file, removeHs=False)

        return MolToSmarts(mol)

    @staticmethod
    def get_mol(pdb_file):
        """
        Use RDKit to generate a mol file
        :param pdb_file: The molecule input file.
        :return: The name of the mol file made.
        """

        mol = MolFromPDBFile(pdb_file, removeHs=False)

        mol_name = pdb_file[:-4] + '.mol'
        MolToMolFile(mol, mol_name)

        return mol_name
