from copy import deepcopy
from typing import List, Tuple

from openff.toolkit.topology import (
    Molecule,
    Topology,
    TopologyAtom,
    TopologyVirtualSite,
    VirtualSite,
)
from openff.toolkit.typing.engines.smirnoff.parameters import (
    ChargeIncrementModelHandler,
    ElectrostaticsHandler,
    LibraryChargeHandler,
    ParameterAttribute,
    ParameterType,
    ToolkitAM1BCCHandler,
    VirtualSiteHandler,
    _allow_only,
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

    hfree = ParameterAttribute(0 * unit.angstroms, unit=unit.angstroms)
    xfree = ParameterAttribute(0 * unit.angstroms, unit=unit.angstroms)
    cfree = ParameterAttribute(0 * unit.angstroms, unit=unit.angstroms)
    nfree = ParameterAttribute(0 * unit.angstroms, unit=unit.angstroms)
    ofree = ParameterAttribute(0 * unit.angstroms, unit=unit.angstroms)
    clfree = ParameterAttribute(0 * unit.angstroms, unit=unit.angstroms)
    sfree = ParameterAttribute(0 * unit.angstroms, unit=unit.angstroms)
    ffree = ParameterAttribute(0 * unit.angstroms, unit=unit.angstroms)
    brfree = ParameterAttribute(0 * unit.angstroms, unit=unit.angstroms)
    pfree = ParameterAttribute(0 * unit.angstroms, unit=unit.angstroms)
    ifree = ParameterAttribute(0 * unit.angstroms, unit=unit.angstroms)
    bfree = ParameterAttribute(0 * unit.angstroms, unit=unit.angstroms)
    sifree = ParameterAttribute(0 * unit.angstroms, unit=unit.angstroms)
    alpha = ParameterAttribute(1)
    beta = ParameterAttribute(0)

    class QUBEKitvdWType(ParameterType):
        """A dummy vdw type which just stores the volume and so we can build the system correctly"""

        _VALENCE_TYPE = "Atom"  # ChemicalEnvironment valence type expected for SMARTS
        _ELEMENT_NAME = "Atom"

        volume = ParameterAttribute(unit=unit.bohr**3)

    _TAGNAME = "QUBEKitvdWTS"
    _INFOTYPE = QUBEKitvdWType
    _DEPENDENCIES = None  # we might need to depend on vdW if present

    def create_force(self, system, topology, **kwargs):
        """over write the force creation to use qubekit"""
        # Get the OpenMM Nonbonded force or add if missing
        existing = [system.getForce(i) for i in range(system.getNumForces())]
        existing = [f for f in existing if type(f) == self._OPENMMTYPE]

        # if not present make one and add the particles
        if len(existing) == 0:
            force = self._OPENMMTYPE()
            system.addForce(force)
            # add all atom particles, Vsites are added later
            for _ in topology.topology_atoms:
                force.addParticle(0.0, 1.0, 0.0)
        else:
            force = existing[0]

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
                "H": h_base(r_free=self.hfree.value_in_unit(unit.angstroms)),
                "C": c_base(r_free=self.cfree.value_in_unit(unit.angstroms)),
                "X": h_base(r_free=self.xfree.value_in_unit(unit.angstroms)),
                "O": o_base(r_free=self.ofree.value_in_unit(unit.angstroms)),
                "N": n_base(r_free=self.nfree.value_in_unit(unit.angstroms)),
                "Cl": cl_base(r_free=self.clfree.value_in_unit(unit.angstroms)),
                "S": s_base(r_free=self.sfree.value_in_unit(unit.angstroms)),
                "F": f_base(r_free=self.ffree.value_in_unit(unit.angstroms)),
                "Br": br_base(r_free=self.brfree.value_in_unit(unit.angstroms)),
                "I": i_base(r_free=self.ifree.value_in_unit(unit.angstroms)),
                "P": p_base(r_free=self.ifree.value_in_unit(unit.angstroms)),
                "B": b_base(r_free=self.bfree.value_in_unit(unit.angstroms)),
                "Si": si_base(r_free=self.sifree.value_in_unit(unit.angstroms)),
            },
            alpha=self.alpha,
            beta=self.beta,
        )

        water = Molecule.from_smiles("O")

        for ref_mol in topology.reference_molecules:
            # skip any waters
            if ref_mol == water:
                continue

            # if the molecule has no conformer generate one
            if ref_mol.n_conformers == 0:
                ref_mol.generate_conformers(n_conformers=1)

            qb_mol = Ligand.from_rdkit(ref_mol.to_rdkit())
            for parameter in self.parameters:
                matches = ref_mol.chemical_environment_matches(
                    parameter.smirks, unique=False
                )
                for match in matches:
                    qb_mol.atoms[match[0]].aim.volume = parameter.volume.value_in_unit(
                        unit.angstroms**3
                    )
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
                        particle_parameters.charge,  # this is a dummy charge which needs to be corrected by a libray charge
                        particle_parameters.sigma,
                        particle_parameters.epsilon,
                    )


class QUBEKitvdWHandler(vdWHandler):
    """
    A subclass of the normal vdWhandler to use for qubekit optimisations so we can mix water models with our custom handler
    """

    _TAGNAME = "QUBEKitvdW"

    def create_force(self, system, topology, **kwargs):
        # Get the OpenMM Nonbonded force or add if missing
        existing = [system.getForce(i) for i in range(system.getNumForces())]
        existing = [f for f in existing if type(f) == self._OPENMMTYPE]

        # if not present make one and add the particles
        if len(existing) == 0:
            force = self._OPENMMTYPE()
            system.addForce(force)
            # add all atom particles, Vsites are added later
            for _ in topology.topology_atoms:
                force.addParticle(0.0, 1.0, 0.0)
        else:
            force = existing[0]

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

        # Iterate over all defined Lennard-Jones types, allowing later matches to override earlier ones.
        atom_matches = self.find_matches(topology)

        # Set the particle Lennard-Jones terms.
        for atom_key, atom_match in atom_matches.items():
            atom_idx = atom_key[0]
            ljtype = atom_match.parameter_type
            if ljtype.sigma is None:
                sigma = 2.0 * ljtype.rmin_half / (2.0 ** (1.0 / 6.0))
            else:
                sigma = ljtype.sigma
            force.setParticleParameters(atom_idx, 0.0, sigma, ljtype.epsilon)


class LocalVirtualSite(VirtualSite):
    """A particle to represent a local virtual site type"""

    def __init__(
        self,
        p1: unit.Quantity,
        p2: unit.Quantity,
        p3: unit.Quantity,
        name: str,
        orientations: List[Tuple[int, ...]],
    ):

        super().__init__(name=name, orientations=orientations)
        self._p1 = p1.in_units_of(unit.nanometer)
        self._p2 = p2.in_units_of(unit.nanometer)
        self._p3 = p3.in_units_of(unit.nanometer)

    @property
    def p1(self):
        return self._p1

    @property
    def p2(self):
        return self._p2

    @property
    def p3(self):
        return self._p3

    def to_dict(self):
        vsite_dict = super().to_dict()
        vsite_dict["p1"] = self._p1
        vsite_dict["p2"] = self._p2
        vsite_dict["p3"] = self._p3
        vsite_dict["vsite_type"] = self.type
        return vsite_dict

    @classmethod
    def from_dict(cls, vsite_dict):
        base_dict = deepcopy(vsite_dict)
        assert vsite_dict["vsite_type"] == "LocalVirtualSite"
        del base_dict["vsite_type"]
        return cls(**base_dict)

    @property
    def local_frame_weights(self):
        o_weights = [1.0, 0.0, 0.0]
        x_weights = [-1.0, 1.0, 0.0]
        y_weights = [-1.0, 0.0, 1.0]

        return o_weights, x_weights, y_weights

    @property
    def local_frame_position(self):
        return [
            self._p1.value_in_unit(unit.nanometer),
            self._p2.value_in_unit(unit.nanometer),
            self._p3.value_in_unit(unit.nanometer),
        ] * unit.nanometer

    def get_openmm_virtual_site(self, atoms: Tuple[int, ...]):
        assert len(atoms) == 3
        return self._openmm_virtual_site(atoms)

    @property
    def type(self) -> str:
        """Hack to work around the use of the default vsite handler to get the parent index"""
        return "DivalentLonePairVirtualSite"


class LocalCoordinateVirtualSiteHandler(VirtualSiteHandler):
    """
    A custom handler to add QUBEKit vsites to openmm systems made via the openff-toolkit.

    Our v-sites are all based on the LocalCoordinateSite and can be translated directly to this object in OpenMM.
    """

    class VirtualSiteType(vdWHandler.vdWType):

        _VALENCE_TYPE = None
        _ELEMENT_NAME = "VirtualSite"

        name = ParameterAttribute(default="EP", converter=str)
        match = ParameterAttribute(default="once", converter=_allow_only(["once"]))
        type = ParameterAttribute(default="local", converter=str)
        x_local = ParameterAttribute(unit=unit.nanometers)
        y_local = ParameterAttribute(unit=unit.nanometers)
        z_local = ParameterAttribute(unit=unit.nanometers)
        charge = ParameterAttribute(unit=unit.elementary_charge)
        sigma = ParameterAttribute(unit=unit.nanometer)
        epsilon = ParameterAttribute(unit.kilojoule_per_mole)

        @property
        def parent_index(self) -> int:
            """The parent is always the first index in a qubekit vsite"""
            return 0

        def to_openmm_particle(
            self, particle_indices: Tuple[int, ...]
        ) -> openmm.LocalCoordinatesSite:
            """Create an openmm local coord site based on the predefined weights using in QUBEKit"""
            if len(particle_indices) == 4:
                o_weights = [1.0, 0.0, 0.0, 0.0]
                x_weights = [-1.0, 0.33333333, 0.33333333, 0.33333333]
                y_weights = [1.0, -1.0, 0.0, 0.0]
            else:
                o_weights = [1.0, 0.0, 0.0]
                x_weights = [-1.0, 1.0, 0.0]
                y_weights = [-1.0, 0.0, 1.0]
            return openmm.LocalCoordinatesSite(
                *particle_indices,
                openmm.Vec3(*o_weights),
                openmm.Vec3(*x_weights),
                openmm.Vec3(*y_weights),
                openmm.Vec3(self.x_local, self.y_local, self.z_local),
            )

        def to_openff_particle(self, orientations: List[Tuple[int, ...]]):
            values_dict = {"p1": self.x_local, "p2": self.y_local, "p3": self.z_local}
            return LocalVirtualSite(
                name=self.name, orientations=orientations, **values_dict
            )

    _TAGNAME = "LocalCoordinateVirtualSites"
    _INFOTYPE = VirtualSiteType
    _OPENMMTYPE = openmm.NonbondedForce
    _DEPENDENCIES = [
        ElectrostaticsHandler,
        LibraryChargeHandler,
        ChargeIncrementModelHandler,
        ToolkitAM1BCCHandler,
        vdWHandler,
    ]

    exclusion_policy = ParameterAttribute(default="parents")

    def create_force(self, system: openmm.System, topology: Topology, **kwargs):

        if system.getNumParticles() != topology.n_topology_atoms:
            raise ValueError("the system does not seem to have any particles in it")

        force = super(VirtualSiteHandler, self).create_force(system, topology, **kwargs)

        matches_by_parent = self._find_matches_by_parent(topology)

        parameter: LocalCoordinateVirtualSiteHandler.VirtualSiteType

        for parent_index, parameters in matches_by_parent.items():
            for parameter, orientations in parameters:
                for orientation in orientations:
                    orientation_indices = orientation.topology_atom_indices
                    openmm_particle = parameter.to_openmm_particle(orientation_indices)

                    charge = parameter.charge
                    sigma = parameter.sigma
                    epsilon = parameter.epsilon

                    # add the vsite with no mass
                    index_system = system.addParticle(0.0)
                    index_force = force.addParticle(charge, sigma, epsilon)
                    assert index_system == index_force

                    system.setVirtualSite(index_system, openmm_particle)

        self._create_openff_virtual_sites(matches_by_parent)
