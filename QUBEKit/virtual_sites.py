from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit
from numpy import sqrt


class VSiteTester:
    """Test the OPLS and virtual site implementation in openmm"""

    def __init__(self, pdb_file, xml_file):
        self.pdb = pdb_file
        self.xml = xml_file
        self.system = None
        self.opls = True
        self.openmm_system()

    def openmm_system(self):
        """Initialise the OpenMM system we will use to evaluate the energies."""

        # Load the initial coords into the system and initialise
        pdb = app.PDBFile(self.pdb)
        forcefield = app.ForceField(self.xml)
        modeller = app.Modeller(pdb.topology, pdb.positions)  # set the initial positions from the pdb
        # now we need to check if there are extra sites and add them to the system
        print(list(modeller.topology.bonds()))
        modeller.addExtraParticles(forcefield)
        # now we can make sure that the particles have been added
        app.PDBFile.writeFile(modeller.topology, modeller.positions, open('modeller.pdb', 'w+'))
        # before setting up the system we need to make sure the modeller.topology includes a parent-site bond

        # for atom in modeller.topology.atoms():
        #     print(atom)
        # atoms = [atom for atom in modeller.topology.atoms()]
        # modeller.topology.addBond(atoms[3], atoms[11])
        for bond in modeller.topology.bonds():
            print(bond)
        data = forcefield._SystemData()
        print(data.atoms)
        print(data.impropers)

        exit()

        self.system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None)

        if self.opls:
            print('using opls rules')
            self.opls_lj()

        exit()

        temperature = 298.15 * unit.kelvin
        integrator = mm.LangevinIntegrator(temperature, 5 / unit.picoseconds, 0.001 * unit.picoseconds)

        self.simulation = app.Simulation(modeller.topology, self.system, integrator)
        self.simulation.context.setPositions(modeller.positions)

    def opls_lj(self, excep_pairs=None, normal_pairs=None):
        """This function changes the standard OpenMM combination rules to use OPLS, execp and normal pairs are only
        required if their are virtual sites in the molecule.
        """

        # Get the system information from the openmm system
        forces = {self.system.getForce(index).__class__.__name__: self.system.getForce(index) for index in
                  range(self.system.getNumForces())}
        # Use the nondonded_force to get the same rules
        nonbonded_force = forces['NonbondedForce']
        lorentz = mm.CustomNonbondedForce(
            'epsilon*((sigma/r)^12-(sigma/r)^6); sigma=sqrt(sigma1*sigma2); epsilon=sqrt(epsilon1*epsilon2)*4.0')
        lorentz.setNonbondedMethod(nonbonded_force.getNonbondedMethod())
        lorentz.addPerParticleParameter('sigma')
        lorentz.addPerParticleParameter('epsilon')
        lorentz.setCutoffDistance(nonbonded_force.getCutoffDistance())
        self.system.addForce(lorentz)

        l_j_set = {}
        # For each particle, collect the particle parameters into the lj method
        for index in range(nonbonded_force.getNumParticles()):
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(index)
            # collect the lj params for the custom method
            l_j_set[index] = (sigma, epsilon)
            lorentz.addParticle([sigma, epsilon])
            # set the standard nonbonded lj parameters to 0 so they are not double counted
            nonbonded_force.setParticleParameters(index, charge, sigma, epsilon * 0)

        for i in range(nonbonded_force.getNumExceptions()):
            (p1, p2, q, sig, eps) = nonbonded_force.getExceptionParameters(i)
            print(p1, p2, q, sig, eps)
            # ALL THE 12,13 and 14 interactions are EXCLUDED FROM CUSTOM NONBONDED FORCE
            lorentz.addExclusion(p1, p2)
            if eps._value != 0.0:
                sig14 = sqrt(l_j_set[p1][0] * l_j_set[p2][0])
                # TODO eps14 not used
                eps14 = sqrt(l_j_set[p1][1] * l_j_set[p2][1])
                nonbonded_force.setExceptionParameters(i, p1, p2, q, sig14, eps)
            # # If there is a virtual site in the molecule we have to change the exceptions and pairs lists
            # # Old method which needs updating
            # if excep_pairs:
            #     for x in range(len(excep_pairs)):  # scale 14 interactions
            #         if p1 == excep_pairs[x, 0] and p2 == excep_pairs[x, 1] or p2 == excep_pairs[x, 0] and p1 == \
            #                 excep_pairs[x, 1]:
            #             charge1, sigma1, epsilon1 = nonbonded_force.getParticleParameters(p1)
            #             charge2, sigma2, epsilon2 = nonbonded_force.getParticleParameters(p2)
            #             q = charge1 * charge2 * 0.5
            #             sig14 = sqrt(sigma1 * sigma2) * 0.5
            #             eps = sqrt(epsilon1 * epsilon2) * 0.5
            #             nonbonded_force.setExceptionParameters(i, p1, p2, q, sig14, eps)
            #
            # if normal_pairs:
            #     for x in range(len(normal_pairs)):
            #         if p1 == normal_pairs[x, 0] and p2 == normal_pairs[x, 1] or p2 == normal_pairs[x, 0] and p1 == \
            #                 normal_pairs[x, 1]:
            #             charge1, sigma1, epsilon1 = nonbonded_force.getParticleParameters(p1)
            #             charge2, sigma2, epsilon2 = nonbonded_force.getParticleParameters(p2)
            #             q = charge1 * charge2
            #             sig14 = sqrt(sigma1 * sigma2)
            #             eps = sqrt(epsilon1 * epsilon2)
            #             nonbonded_force.setExceptionParameters(i, p1, p2, q, sig14, eps)

        return self.system


def main():
    tester = VSiteTester('pyridine.pdb', 'pyridine.xml')


if __name__ == '__main__':
    main()
