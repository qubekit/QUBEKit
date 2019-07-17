#!/usr/bin/env python3

from collections import OrderedDict
from copy import deepcopy

import xml.etree.ElementTree as ET


class Parametrisation:
    """
    Class of methods which perform the initial parametrisation for the molecule.
    The Parameters will be stored into the molecule as dictionaries as this is easy to manipulate and convert
    to a parameter tree.

    Note all parameters gathered here are indexed from 0,
    whereas the ligand object indices start from 1 for all networkx related properties such as bonds!


    Parameters
    ---------
    molecule : QUBEKit molecule object

    input_file : an OpenMM style xml file associated with the molecule object

    fftype : the FF type the molecule will be parametrised with
             only needed in the case of gaff or gaff2 else will be assigned based on class used.

    Returns
    -------
    AtomTypes : dictionary of the atom names, the associated OPLS type and class type stored under number.
                {0: [C00, OPLS_800, C800]}

    Residues : dictionary of residue names indexed by the order they appear.

    HarmonicBondForce : dictionary of equilibrium distances and force constants stored under the bond tuple.
                        {(0, 1): [eqr=456, fc=984375]}

    HarmonicAngleForce : dictionary of equilibrium  angles and force constant stored under the angle tuple.

    PeriodicTorsionForce : dictionary of periodicity, barrier and phase stored under the torsion tuple.

    NonbondedForce : dictionary of charge, sigma and epsilon stored under the original atom ordering.
    """

    def __init__(self, molecule, input_file=None, fftype=None):

        self.molecule = molecule
        self.input_file = input_file
        self.fftype = fftype
        self.molecule.combination = 'opls'
        self.combination = 'amber'

        # could be a problem for boron compounds
        self.molecule.AtomTypes = {}
        self.molecule.HarmonicBondForce = {bond: ['0', '0'] for bond in self.molecule.bond_lengths.keys()}
        self.molecule.HarmonicAngleForce = {angle: ['0', '0'] for angle in self.molecule.angle_values.keys()}
        self.molecule.NonbondedForce = OrderedDict((number, ['0', '0', '0']) for number in range(len(self.molecule.atoms)))
        self.molecule.PeriodicTorsionForce = OrderedDict()

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__!r})'

    def gather_parameters(self):
        """
        This method parses the serialised xml file and collects the parameters ready to pass them
        to build tree.
        """

        # Try to gather the AtomTypes first
        for atom in self.molecule.atoms:
            self.molecule.AtomTypes[atom.atom_index] = [atom.atom_name, 'QUBE_' + str(000 + atom.atom_index),
                                                        str(atom.element) + str(000 + atom.atom_index)]

        try:
            in_root = ET.parse('serialised.xml').getroot()

            # Extract all bond data
            for Bond in in_root.iter('Bond'):
                bond = (int(Bond.get('p1')), int(Bond.get('p2')))
                if bond in self.molecule.HarmonicBondForce:
                    self.molecule.HarmonicBondForce[bond] = [Bond.get('d'), Bond.get('k')]
                else:
                    self.molecule.HarmonicBondForce[bond[::-1]] = [Bond.get('d'), Bond.get('k')]

            # Extract all angle data
            for Angle in in_root.iter('Angle'):
                angle = int(Angle.get('p1')), int(Angle.get('p2')), int(Angle.get('p3'))
                if angle in self.molecule.HarmonicAngleForce:
                    self.molecule.HarmonicAngleForce[angle] = [Angle.get('a'), Angle.get('k')]
                else:
                    self.molecule.HarmonicAngleForce[angle[::-1]] = [Angle.get('a'), Angle.get('k')]

            # Extract all non-bonded data
            i = 0
            for Atom in in_root.iter('Particle'):
                if "eps" in Atom.attrib:
                    self.molecule.NonbondedForce[i] = [Atom.get('q'), Atom.get('sig'), Atom.get('eps')]
                    i += 1

            # Extract all of the torsion data
            phases = ['0', '3.141592653589793', '0', '3.141592653589793']
            for Torsion in in_root.iter('Torsion'):
                tor_str_forward = tuple(int(Torsion.get(f'p{i}')) for i in range(1, 5))
                tor_str_back = tuple(reversed(tor_str_forward))

                if tor_str_forward not in self.molecule.PeriodicTorsionForce and tor_str_back not in self.molecule.PeriodicTorsionForce:
                    self.molecule.PeriodicTorsionForce[tor_str_forward] = [
                        [Torsion.get('periodicity'), Torsion.get('k'), phases[int(Torsion.get('periodicity')) - 1]]]

                elif tor_str_forward in self.molecule.PeriodicTorsionForce:
                    self.molecule.PeriodicTorsionForce[tor_str_forward].append(
                        [Torsion.get('periodicity'), Torsion.get('k'), phases[int(Torsion.get('periodicity')) - 1]])

                elif tor_str_back in self.molecule.PeriodicTorsionForce:
                    self.molecule.PeriodicTorsionForce[tor_str_back].append(
                        [Torsion.get('periodicity'), Torsion.get('k'), phases[int(Torsion.get('periodicity')) - 1]])

        except FileNotFoundError:
            # Check what parameter engine we are using if not none then raise an error
            if self.molecule.parameter_engine != 'none':
                raise FileNotFoundError('Molecule could not be serialised from openMM')
        # Now we have all of the torsions from the openMM system
        # we should check if any torsions we found in the molecule do not have parameters
        # if they don't give them the default 0 parameter this will not change the energy
        if self.molecule.dihedrals is not None:
            for tor_list in self.molecule.dihedrals.values():
                for torsion in tor_list:
                    if torsion not in self.molecule.PeriodicTorsionForce and tuple(reversed(torsion)) not in self.molecule.PeriodicTorsionForce:
                        self.molecule.PeriodicTorsionForce[torsion] = [['1', '0', '0'], ['2', '0', '3.141592653589793'],
                                                                       ['3', '0', '0'], ['4', '0', '3.141592653589793']]
        torsions = [sorted(key) for key in self.molecule.PeriodicTorsionForce.keys()]
        if self.molecule.improper_torsions is not None:
            for torsion in self.molecule.improper_torsions:
                if torsion not in torsions:
                    # The improper torsion is missing and should be added with no energy
                    self.molecule.PeriodicTorsionForce[torsion] = [['1', '0', '0'], ['2', '0', '3.141592653589793'],
                                                                   ['3', '0', '0'], ['4', '0', '3.141592653589793']]

        # Now we need to fill in all blank phases of the Torsions
        for key, val in self.molecule.PeriodicTorsionForce.items():
            vns = ['1', '2', '3', '4']
            if len(val) < 4:
                # now need to add the missing terms from the torsion force
                for force in val:
                    vns.remove(force[0])
                for i in vns:
                    val.append([i, '0', phases[int(i) - 1]])
        # sort by periodicity using lambda function
        for val in self.molecule.PeriodicTorsionForce.values():
            val.sort(key=lambda x: x[0])

        # now we need to tag the proper and improper torsions and reorder them so the first atom is the central
        improper_torsions = None
        if self.molecule.improper_torsions is not None:
            improper_torsions = OrderedDict()
            for improper in self.molecule.improper_torsions:
                for key, val in self.molecule.PeriodicTorsionForce.items():
                    # for each improper find the corresponding torsion parameters and save
                    if sorted(key) == sorted(improper):
                        # if they match tag the dihedral
                        self.molecule.PeriodicTorsionForce[key].append('Improper')
                        # replace the key with the strict improper order first atom is center
                        improper_torsions[improper] = val

        torsions = deepcopy(self.molecule.PeriodicTorsionForce)
        # Remake the torsion; store in the ligand
        self.molecule.PeriodicTorsionForce = OrderedDict((v, k) for v, k in torsions.items() if k[-1] != 'Improper')

        # Add the impropers at the end of the torsion object
        if improper_torsions is not None:
            for key, val in improper_torsions.items():
                self.molecule.PeriodicTorsionForce[key] = val

    def get_symmetry(self):
        """
        Using the initial FF parameters cluster the bond and angle terms
        :return: Dictionaries that can be used to cluster the parameters latter on after the seminaro method
        """

        bond_types = {}
        angle_types = {}

        if next(iter(self.molecule.HarmonicBondForce.values())) == ['0', '0']:
            return

        for bond, value in sorted(self.molecule.HarmonicBondForce.items()):
            bond_types.setdefault(tuple(value), []).append(bond)

        for angle, value in sorted(self.molecule.HarmonicAngleForce.items()):
            angle_types.setdefault(tuple(value), []).append(angle)

        # Now store back into the ligand
        if bond_types:
            self.molecule.bond_types = bond_types
        if angle_types:
            self.molecule.angle_types = angle_types
