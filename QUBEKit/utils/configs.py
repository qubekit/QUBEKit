#!/usr/bin/env python3

from QUBEKit.utils.helpers import string_to_bool
from pydantic import BaseModel, Field, PositiveInt

from typing import Optional
from typing_extensions import Literal

from configparser import ConfigParser
import os


class ConfigBase(BaseModel):

    class Config:
        validate_assignment = True


class ParamDummy(ConfigBase):
    parameter_engine: str
    forcefield: str

    class Config:
        # define the inherited fields here, to keep the descriptions
        fields = {
            "forcefield": {"description": "The version of the forcefield that should be used."},
            "parameter_engine": {"description": "The name of the parameter engine that should be used."}
        }


class AnteChamberDummy(ParamDummy):
    """
    Each class would have to be rebased on pydantic to fit into the pipline, this is a dummy class for now.
    """
    parameter_engine: Literal["AnteChamberDummy"] = "AnteChamberDummy"
    forcefield: Literal["gaff"] = "gaff"


class OpenffDummy(ParamDummy):
    parameter_engine: Literal["OpenffDummy"] = "OpenffDummy"
    forcefield: Literal["openff_unconstrained-1.3.0.offxml"] = "openff_unconstrained-1.3.0.offxml"


class GeometryOptimiser(ConfigBase):
    """
    A dummy class to handle generale geometry optimisations, this can handle the dispatch to the correct method.
    """
    #TODO remove this option, as we move to qcengine we have to use geometric or the other optimisers like opt king
    # native optimisation is not an option

    geometric: bool = Field(True, description="If geometric should be used to optimise the molecule")
    program: Literal["openmm", "xtb", "psi4", "gaussian", "torchani", "rdkit"] = Field("openmm", description="The program that should be used to perform the optimisation.")
    method: str = Field("openff_unconstrained-1.3.0.offxml", description="The method that should be used to run the optimisation")
    basis: Optional[str] = Field("smirnoff", description="The basis that should be used during the optimisation.")
    maxiter: PositiveInt = Field(350, description="The maximum number of optimisation steps.")
    convergence: Literal["GAU", "GAU_TIGHT"] = Field("GAU_TIGHT", description="The convergence critera for the geometry optimisation.")


class Workflow(ConfigBase):
    """
    A schema which lists all of the run time options for each stage in the workflow.
    """
    parametrisation: ParamDummy = Field(AnteChamberDummy(), description="The parameterisation method that should be used.")
    mm_minimisation: GeometryOptimiser = Field(GeometryOptimiser(), description="The method that should be used to do an initial optimisation, this could be mm,xtb or ml or cheap QM.")


class GeneralConfig(ConfigBase):
    """
    This schema handles all run time options for QUBEkit, each workflow stage is listed separately with its own schema which is fully validated
    to catch errors when loading the config ahead of run time.
    """

    # general settings
    memory: PositiveInt = Field(4, description="The amount of memory each qm job is allowed to use in GB.")
    threads: PositiveInt = Field(2, description="The number of threads each qm job is allowed to use.")
    workflow: Workflow = Field(Workflow(),
                               description="The parametrisation workflow which details what option should be used for each fitting stage.")


class Configure:
    """
    Class to help load, read and write ini style configuration files returns dictionaries of the config
    settings as strings, all numbers must then be cast before use.
    """

    home = os.path.expanduser('~')
    config_folder = os.path.join(home, 'QUBEKit_configs')
    master_file = 'master_config.ini'

    qm = {
        'theory': 'B3LYP',              # Theory to use in freq and dihedral scans recommended e.g. wB97XD or B3LYP
        'basis': '6-311++G(d,p)',       # Basis set
        'vib_scaling': '1',             # Associated scaling to the theory
        'threads': '2',                 # Number of processors used in Gaussian09; affects the bonds and dihedral scans
        'memory': '2',                  # Amount of memory (in GB); specified in the Gaussian09 scripts
        'convergence': 'GAU_TIGHT',     # Criterion used during optimisations; works using PSI4, GeomeTRIC and G09
        'iterations': '350',            # Max number of optimisation iterations
        'bonds_engine': 'g09',          # Engine used for bonds calculations
        'density_engine': 'g09',        # Engine used to calculate the electron density
        'charges_engine': 'chargemol',  # Engine used for charge partitioning
        'ddec_version': '6',            # DDEC version used by Chargemol, 6 recommended but 3 is also available
        'dielectric': '4.0',            # During density stage, which dielectric constant should be used
        'geometric': 'True',            # Use GeomeTRIC for optimised structure (if False, will just use PSI4)
        'solvent': 'True',              # Use a solvent in the PSI4/Gaussian09 input
        'enable_symmetry': 'True',      # Enable or disable the use of symmetrisation for bond, angle, charge, and Lennard-Jones parameters
        'enable_virtual_sites': 'False',    # Enable or disable the use of virtual sites in the charge step
        'v_site_error_factor': '1.005',     # Maximum error factor from adding a site that means the site will be kept
    }

    fitting = {
        'dih_start': '-165',            # Starting angle of dihedral scan
        'increment': '15',              # Angle increase increment
        'dih_end': '180',               # The last dihedral angle in the scan
        't_weight': 'infinity',         # Weighting temperature that can be changed to better fit complicated surfaces
        'opt_method': 'BFGS',           # The type of SciPy optimiser to use
        'refinement_method': 'SP',      # The type of QUBE refinement that should be done SP: single point energies
        'tor_limit': '20',              # Torsion Vn limit to speed up fitting
        'div_index': '0',               # Fitting starting index in the division array
        'parameter_engine': 'antechamber',      # Method used for initial parametrisation
        'l_pen': '0.0',                 # The regularisation penalty
        'relative_to_global': 'False'   # If we should compute our relative energy surface
                                        # compared to the global minimum
    }

    excited = {
        'excited_state': 'False',       # Is this an excited state calculation
        'excited_theory': 'TDA',        # Excited state theory TDA or TD
        'n_states': '3',
        'excited_root': '1',
        'use_pseudo': 'False',
        'pseudo_potential_block': ''
    }

    descriptions = {
        'chargemol': '/home/<QUBEKit_user>/chargemol_09_26_2017',  # Location of the chargemol program directory
        'log': '999',                   # Default string for the working directories and logs
    }

    help = {
        'theory': ';Theory to use in freq and dihedral scans recommended wB97XD or B3LYP, for example',
        'basis': ';Basis set',
        'vib_scaling': ';Associated scaling to the theory',
        'threads': ';Number of processors used in g09; affects the bonds and dihedral scans',
        'memory': ';Amount of memory (in GB); specified in the g09 and PSI4 scripts',
        'convergence': ';Criterion used during optimisations; GAU, GAU_TIGHT, GAU_VERYTIGHT',
        'iterations': ';Max number of optimisation iterations',
        'bonds_engine': ';Engine used for bonds calculations',
        'density_engine': ';Engine used to calculate the electron density',
        'charges_engine': ';Engine used for charge partitioning',
        'ddec_version': ';DDEC version used by Chargemol, 6 recommended but 3 is also available',
        'dielectric': ';During density stage, which dielectric constant should be used',
        'geometric': ';Use geometric for optimised structure (if False, will just use PSI4)',
        'solvent': ';Use a solvent in the psi4/gaussian09 input',
        'enable_symmetry': ';Enable or disable the use of symmetrisation for bond, angle, charge, and Lennard-Jones parameters',
        'enable_virtual_sites': ';Enable or disable the use of virtual sites in the charge step',
        'v_site_error_factor': ';Maximum error factor from adding a site that means the site will be kept',
        'dih_start': ';Starting angle of dihedral scan',
        'increment': ';Angle increase increment',
        'dih_end': ';The last dihedral angle in the scan',
        't_weight': ';Weighting temperature that can be changed to better fit complicated surfaces',
        'l_pen': ';The regularisation penalty',
        'relative_to_global': ';If we should compute our relative energy surface compared to the global minimum',
        'opt_method': ';The type of SciPy optimiser to use',
        'refinement_method': ';The type of QUBE refinement that should be done SP: single point energies',
        'tor_limit': ';Torsion Vn limit to speed up fitting',
        'div_index': ';Fitting starting index in the division array',
        'parameter_engine': ';Method used for initial parametrisation',
        'chargemol': ';Location of the Chargemol program directory (do not end with a "/")',
        'log': ';Default string for the names of the working directories',
        'excited_state': ';Use the excited state',
        'excited_theory': ';Excited state theory TDA or TD',
        'n_states': ';The number of states to use',
        'excited_root': ';The root',
        'use_pseudo': ';Use a pseudo potential',
        'pseudo_potential_block': ';Enter the pseudo potential block here',
    }

    def load_config(self, config_file='default_config'):
        """
        This method loads and returns the selected config file.
        It also converts strings to ints, bools or floats where appropriate.
        """

        if config_file is None:
            config_file = 'default_config'

        if config_file == 'default_config':

            # Check if the user has made a new master file to use
            if self.check_master():
                qm, fitting, excited, descriptions = self.ini_parser(os.path.join(self.config_folder, self.master_file))

            else:
                # If there is no master then assign the default config
                qm, fitting, excited, descriptions = self.qm, self.fitting, self.excited, self.descriptions

        else:
            # Load in the ini file given
            if os.path.exists(config_file):
                qm, fitting, excited, descriptions = self.ini_parser(config_file)

            else:
                qm, fitting, excited, descriptions = self.ini_parser(os.path.join(self.config_folder, config_file))

        # List of keys whose values should be cast to ints
        for key in ['threads', 'memory', 'iterations', 'ddec_version', 'dih_start',
                    'increment', 'dih_end', 'tor_limit', 'div_index', 'n_states', 'excited_root']:
            if key in qm:
                qm[key] = int(qm[key])
            elif key in fitting:
                fitting[key] = int(fitting[key])
            elif key in excited:
                excited[key] = int(excited[key])

        # Cast the floats
        qm['vib_scaling'] = float(qm['vib_scaling'])
        qm['dielectric'] = float(qm['dielectric'])
        qm['v_site_error_factor'] = float(qm['v_site_error_factor'])
        fitting['l_pen'] = float(fitting['l_pen'])
        if fitting['t_weight'] != 'infinity':
            fitting['t_weight'] = float(fitting['t_weight'])

        # List of keys whose values should be cast to bools
        for key in ['geometric', 'solvent', 'enable_symmetry', 'enable_virtual_sites', 'excited_state',
                    'use_pseudo', 'relative_to_global']:
            if key in qm:
                qm[key] = string_to_bool(qm[key])
            elif key in fitting:
                fitting[key] = string_to_bool(fitting[key])
            elif key in excited:
                excited[key] = string_to_bool(excited[key])

        return {**qm, **fitting, **excited, **descriptions}

    @staticmethod
    def ini_parser(ini):
        """Parse an ini type config file and return the arguments as dictionaries."""

        config = ConfigParser(allow_no_value=True)
        config.read(ini)
        qm = config.__dict__['_sections']['QM']
        fitting = config.__dict__['_sections']['FITTING']
        excited = config.__dict__['_sections']['EXCITED']
        descriptions = config.__dict__['_sections']['DESCRIPTIONS']

        return qm, fitting, excited, descriptions

    def show_ini(self):
        """Show all of the ini file options in the config folder."""

        # Hide the emacs backups
        return [ini for ini in os.listdir(self.config_folder) if not ini.endswith('~')]

    def check_master(self):
        """Check if there is a new master ini file in the configs folder."""

        return os.path.exists(os.path.join(self.config_folder, self.master_file))

    def ini_writer(self, ini):
        """Make a new configuration file in the config folder using the current master as a template."""

        # make sure the ini file has an ini ending
        if not ini.endswith('.ini'):
            ini += '.ini'

        # Set config parser to allow for comments
        config = ConfigParser(allow_no_value=True)

        categories = {
            'QM': self.qm,
            'FITTING': self.fitting,
            'EXCITED': self.excited,
            'DESCRIPTIONS': self.descriptions
        }

        for name, data in categories.items():
            config.add_section(name)
            for key, val in data.items():
                config.set(name, self.help[key])
                config.set(name, key, val)

        with open(os.path.join(self.config_folder, ini), 'w+') as out:
            config.write(out)

    def ini_edit(self, ini_file):
        """
        Open the ini file for editing in the command line using emacs.
        """

        # Make sure the ini file has an ini ending
        if not ini_file.endswith('.ini'):
            ini_file += '.ini'

        ini_path = os.path.join(self.config_folder, ini_file)
        os.system(f'emacs -nw {ini_path}')
