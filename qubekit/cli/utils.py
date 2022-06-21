from openff.toolkit.topology import TopologyAtom, TopologyVirtualSite
from openff.toolkit.typing.engines.smirnoff.parameters import (
    ParameterAttribute,
    ParameterType,
    vdWHandler,
)
from openmm import openmm, unit

from qubekit.molecules import Ligand
from qubekit.nonbonded import LennardJones612
from qubekit.nonbonded.protocols import (
    b_base,
    br_base,
    c_base,
    cl_base,
    f_base,
    h_base,
    i_base,
    n_base,
    o_base,
    p_base,
    s_base,
    si_base,
)


class QUBEKitHandler(vdWHandler):
    """A plugin handler to enable the fitting of Rfree parameters using evaluator"""

    hfree = ParameterAttribute(0, unit=unit.angstroms)
    xfree = ParameterAttribute(0, unit=unit.angstroms)
    cfree = ParameterAttribute(0, unit=unit.angstroms)
    nfree = ParameterAttribute(0, unit=unit.angstroms)
    ofree = ParameterAttribute(0, unit=unit.angstroms)
    clfree = ParameterAttribute(0, unit=unit.angstroms)
    sfree = ParameterAttribute(0, unit=unit.angstroms)
    ffree = ParameterAttribute(0, unit=unit.angstroms)
    brfree = ParameterAttribute(0, unit=unit.angstroms)
    pfree = ParameterAttribute(0, unit=unit.angstroms)
    ifree = ParameterAttribute(0, unit=unit.angstroms)
    bfree = ParameterAttribute(0, unit=unit.angstroms)
    sifree = ParameterAttribute(0, unit=unit.angstroms)
    alpha = ParameterAttribute(1)
    beta = ParameterAttribute(0)

    class QUBEKitvdWType(ParameterType):
        """A dummy vdw type which just stores the volume and so we can build the system correctly"""

        _VALENCE_TYPE = "Atom"  # ChemicalEnvironment valence type expected for SMARTS
        _ELEMENT_NAME = "Atom"

        volume = ParameterAttribute(unit=unit.bohr**3)

    _TAGNAME = "QUBEKitvdW"
    _INFOTYPE = QUBEKitvdWType
    _DEPENDENCIES = [vdWHandler]

    def create_force(self, system, topology, **kwargs):
        """over write the force creation to use qubekit"""
        force = super().create_force(system, topology, **kwargs)

        # If we're using PME, then the only possible openMM Nonbonded type is LJPME
        if self.method == "PME":
            # If we're given a nonperiodic box, we always set NoCutoff. Later we'll add support for CutoffNonPeriodic
            if topology.box_vectors is None:
                force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
                # if (topology.box_vectors is None):
                #     raise SMIRNOFFSpecError("If vdW method is  PME, a periodic Topology "
                #                             "must be provided")
            else:
                force.setNonbondedMethod(openmm.NonbondedForce.LJPME)
                force.setCutoffDistance(self.cutoff)
                force.setEwaldErrorTolerance(1.0e-4)

        # If method is cutoff, then we currently support openMM's PME for periodic system and NoCutoff for nonperiodic
        elif self.method == "cutoff":
            # If we're given a nonperiodic box, we always set NoCutoff. Later we'll add support for CutoffNonPeriodic
            if topology.box_vectors is None:
                force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
            else:
                force.setNonbondedMethod(openmm.NonbondedForce.PME)
                force.setUseDispersionCorrection(True)
                force.setCutoffDistance(self.cutoff)

        lj = LennardJones612(
            free_parameters={
                "H": h_base(r_free=self.hfree),
                "C": c_base(r_free=self.cfree),
                "X": h_base(r_free=self.xfree),
                "O": o_base(r_free=self.ofree),
                "N": n_base(r_free=self.nfree),
                "Cl": cl_base(r_free=self.clfree),
                "S": s_base(r_free=self.sfree),
                "F": f_base(r_free=self.ffree),
                "Br": br_base(r_free=self.brfree),
                "I": i_base(r_free=self.ifree),
                "P": p_base(r_free=self.ifree),
                "B": b_base(r_free=self.bfree),
                "Si": si_base(r_free=self.sifree),
            },
            alpha=self.alpha,
            beta=self.beta,
        )

        for ref_mol in topology.reference_molecules:
            # if the molecule is water skip it
            if ref_mol.n_conformers == 0:
                ref_mol.generate_conformers(n_conformers=1)

            qb_mol = Ligand.from_rdkit(ref_mol.to_rdkit())
            for parameter in self.parameters:
                matches = ref_mol.chemical_environment_matches(
                    parameter.smirks, unique=False
                )
                for match in matches:
                    qb_mol.atoms[match[0]].aim.volume = parameter.volume
            # make sure all atoms in the molecule have volumes, assign dummy values
            for i in range(qb_mol.n_atoms):
                atom = qb_mol.atoms[i]
                assert atom.aim.volume is not None
                qb_mol.NonbondedForce.create_parameter(
                    atoms=(i,), charge=0, sigma=0, epsilon=0
                )

            # calculate the nonbonded terms
            lj.run(qb_mol)

            # assign to all copies in the system
            for topology_molecule in topology._reference_molecule_to_topology_molecules[
                ref_mol
            ]:
                for topology_particle in topology_molecule.atoms:
                    if type(topology_particle) is TopologyAtom:
                        ref_mol_particle_index = (
                            topology_particle.atom.molecule_particle_index
                        )
                    elif type(topology_particle) is TopologyVirtualSite:
                        ref_mol_particle_index = (
                            topology_particle.virtual_site.molecule_particle_index
                        )
                    else:
                        raise ValueError(
                            f"Particles of type {type(topology_particle)} are not supported"
                        )

                    topology_particle_index = topology_particle.topology_particle_index
                    particle_parameters = qb_mol.NonbondedForce[
                        (ref_mol_particle_index,)
                    ]
                    # Set the nonbonded force parameters
                    force.setParticleParameters(
                        topology_particle_index,
                        particle_parameters.charge,  # this is a dummy charge which needs to be corrected
                        particle_parameters.sigma,
                        particle_parameters.epsilon,
                    )
