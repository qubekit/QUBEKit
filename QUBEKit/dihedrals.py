#!/usr/bin/env python

from subprocess import call as sub_call


class TorsionScan:
    """This class will take a QUBEKit molecule object and perform a torsiondrive QM (and MM if True) energy scan
    for each selected dihedral.
    """

    def __init__(self, molecule, QMengine, MMengine='openmm', MM_scan=True, native_opt=False, verbose=False):
        self.QMengine = QMengine
        self.MMengine = MMengine
        self.MM_scan = MM_scan
        self.constraints = None
        self.grid_space = QMengine.fitting['increment']
        self.native_opt = native_opt
        self.verbose = verbose
        self.scan_mol = molecule
        self.cmd = ''
        self.find_scan_order()
        self.torsion_cmd()

    def find_scan_order(self):
        """Function takes the molecule and displays the rotatable central bonds,
        the user then enters the number of the torsions to be scanned in the order to be scanned.
        The molecule can also be supplied with a scan order already.
        """

        if self.scan_mol.scan_order:
            return self.scan_mol

        elif len(self.scan_mol.rotatable) == 1:
            print('One rotatable torsion found')
            self.scan_mol.scan_order = self.scan_mol.rotatable
            return self.scan_mol

        elif len(self.scan_mol.rotatable) == 0:
            print('No rotatable torsions found in the molecule')
            self.scan_mol.scan_order = []
            return self.scan_mol

        else:
            # get the rotatable dihedrals from the molecule
            rotatable = list(self.scan_mol.rotatable)
            print('Please select the central bonds round which you wish to scan')
            print('Torsion number   Central-Bond   Representative Dihedral')
            for i, bond in enumerate(rotatable):
                print('  {}                    {}-{}             {}-{}-{}-{}'.format(i + 1, bond[0], bond[1], self.scan_mol.atom_names[self.scan_mol.dihedrals[bond][0][0]-1], self.scan_mol.atom_names[self.scan_mol.dihedrals[bond][0][1]-1], self.scan_mol.atom_names[self.scan_mol.dihedrals[bond][0][2]-1], self.scan_mol.atom_names[self.scan_mol.dihedrals[bond][0][3]-1]))
            scans = list(input('>'))  # Enter as a space separated list
            scans[:] = [scan for scan in scans if scan != ' ']  # remove all spaces from the scan list
            print(scans)

            scan_order = []
            # now add the rotatable dihedral keys to an array
            for scan in scans:
                scan_order.append(rotatable[int(scan)-1])
            self.scan_mol.scan_order = scan_order

            return self.scan_mol

    def QM_scan_input(self, scan):
        """Function takes the rotatable dihedrals requested and writes a scan input file for torsiondrive."""

        with open('dihedrals.txt', 'w+') as out:

            out.write('# dihedral definition by atom indices starting from 0\n# i     j     k     l\n')
            out.write('  {}     {}     {}     {}\n'.format(self.scan_mol.dihedrals[scan][0][0], self.scan_mol.dihedrals[scan][0][1], self.scan_mol.dihedrals[scan][0][2], self.scan_mol.dihedrals[scan][0][3]))
        # TODO need to add PSI4 redundant mode selector
        if self.native_opt:
            self.QMengine.generate_input(optimize=True, threads=True)

        else:
            self.QMengine.geo_gradient(run=False, threads=True)

    def torsion_cmd(self):
        """Function generates a command string to run torsiondrive based on the input commands."""

        # add the first basic command elements
        self.cmd += f'torsiondrive-launch {self.scan_mol.name}.{self.QMengine.__class__.__name__.lower()}in dihedrals.txt '
        if self.grid_space:
            self.cmd += f'-g {self.grid_space} '
        if self.QMengine:
            self.cmd += f'-e {self.QMengine.__class__.__name__.lower()} '
        if self.native_opt:
            self.cmd += '--native_opt '
        if self.verbose:
            self.cmd += '-v '
        return self.cmd

    def get_energy(self, scan):
        """Function will extract an array of energies from the scan results
        and store it back into the molecule in a dictionary using the scan order as keys.
        """

        with open('scan.xyz', 'r') as scan_file:
            scan_energy = []
            for line in scan_file:
                if 'Energy ' in line:
                    scan_energy.append(float(line.split()[3]))

            self.scan_mol.QM_scan_energy[scan] = scan_energy

            return self.scan_mol

    def start_scan(self):
        """Function makes a folder and writes a new a dihedral input file for each scan."""
        # TODO put all the scans in a work queue so they can be performed in parallel

        from os import mkdir, chdir

        for scan in self.scan_mol.scan_order:
            try:
                mkdir(f'SCAN_{scan}')
            except:
                raise Exception(f'Cannot create SCAN_{scan} dir.')
            chdir(f'SCAN_{scan}')
            # now make the scan input files
            self.QM_scan_input(scan)
            sub_call(self.cmd, shell=True)
            self.get_energy(scan)


class TorsionOptimizer:
    """Torsion optimizer class used to optimize dihedral parameters with a range of methods"""
    # TODO add objective function
    # TODO collect the dihedral angles
    # TODO write torsion energy contribution function
    # TODO add method selection, do wide scan then clean up energy surface
    # TODO plot the final energies
    def __init__(self):
        pass
