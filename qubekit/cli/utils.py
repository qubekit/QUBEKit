from copy import deepcopy
from typing import Dict, List, Tuple

from openff.toolkit.topology import (
    Molecule,
    Topology,
    TopologyAtom,
    TopologyVirtualSite,
    VirtualSite,
)
from openff.toolkit.typing.engines.smirnoff.parameters import (
    AngleHandler,
    BondHandler,
    ChargeIncrementModelHandler,
    ConstraintHandler,
    ElectrostaticsHandler,
    IndexedParameterAttribute,
    LibraryChargeHandler,
    ParameterAttribute,
    ParameterHandler,
    ParameterType,
    ToolkitAM1BCCHandler,
    VirtualSiteHandler,
    _allow_only,
    vdWHandler,
)
from openff.toolkit.utils.exceptions import NotBondedError, SMIRNOFFSpecError
from openff.toolkit.utils.toolkits import GLOBAL_TOOLKIT_REGISTRY
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


def _get_nonbonded_force(
    system: openmm.System, topology: Topology
) -> openmm.NonbondedForce:
    """A workaround calling to the super methods of plugins to get the nonbonded force from an openmm system."""
    # Get the OpenMM Nonbonded force or add if missing
    existing = [system.getForce(i) for i in range(system.getNumForces())]
    existing = [f for f in existing if type(f) == openmm.NonbondedForce]

    # if not present make one and add the particles
    if len(existing) == 0:
        force = openmm.NonbondedForce()
        system.addForce(force)
        # add all atom particles, Vsites are added later
        for _ in topology.topology_atoms:
            force.addParticle(0.0, 1.0, 0.0)
    else:
        force = existing[0]
    return force


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
    lj_on_polar_h = ParameterAttribute(
        default="True", converter=_allow_only(["True", "False"])
    )

    class QUBEKitvdWType(ParameterType):
        """A dummy vdw type which just stores the volume and so we can build the system correctly"""

        _VALENCE_TYPE = "Atom"  # ChemicalEnvironment valence type expected for SMARTS
        _ELEMENT_NAME = "Atom"

        name = ParameterAttribute(default=None)
        volume = IndexedParameterAttribute(unit=unit.bohr**3)

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            unique_tags, connectivity = GLOBAL_TOOLKIT_REGISTRY.call(
                "get_tagged_smarts_connectivity", self.smirks
            )
            if len(self.volume) != len(unique_tags):
                raise SMIRNOFFSpecError(
                    f"QUBEKitHandler {self} was initialized with unequal number of "
                    f"tagged atoms and volumes"
                )

    _TAGNAME = "QUBEKitvdWTS"
    _INFOTYPE = QUBEKitvdWType
    # vdW must go first as we need to overwrite the blank parameters
    _DEPENDENCIES = [
        vdWHandler,
        ElectrostaticsHandler,
    ]  # we might need to depend on vdW if present

    def create_force(self, system, topology, **kwargs):
        """over write the force creation to use qubekit"""
        force = _get_nonbonded_force(system=system, topology=topology)

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
            lj_on_polar_h=self.lj_on_polar_h,
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
            # each parameter should cover the whole molecule
            for parameter in self.parameters:
                matches = ref_mol.chemical_environment_matches(
                    parameter.smirks, unique=True
                )
                if matches and len(matches[0]) != qb_mol.n_atoms:
                    raise SMIRNOFFSpecError(
                        f"Parameter {parameter.smirks} matched with {ref_mol} but the whole molecule was not covered!"
                    )
                if matches:
                    for atom in matches[0]:
                        qb_mol.atoms[atom].aim.volume = parameter.volume[
                            atom
                        ].value_in_unit(unit.bohr**3)
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
                    # get the current particle charge as we do not want to change this
                    charge, _, _ = force.getParticleParameters(topology_particle_index)
                    # Set the nonbonded force parameters
                    force.setParticleParameters(
                        topology_particle_index,
                        charge,  # set with the existing charge do not use the dummy qubekit value!
                        particle_parameters.sigma,
                        particle_parameters.epsilon,
                    )


class LocalVirtualSite(VirtualSite):
    """A particle to represent a local virtual site type"""

    def __init__(
        self,
        p1: unit.Quantity,
        p2: unit.Quantity,
        p3: unit.Quantity,
        name: str,
        o_weights: List[float],
        x_weights: List[float],
        y_weights: List[float],
        orientations: List[Tuple[int, ...]],
    ):
        super().__init__(name=name, orientations=orientations)
        self._p1 = p1.in_units_of(unit.nanometer)
        self._p2 = p2.in_units_of(unit.nanometer)
        self._p3 = p3.in_units_of(unit.nanometer)
        self._o_weights = o_weights
        self._x_weights = x_weights
        self._y_weights = y_weights

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
        vsite_dict["o_weights"] = self._o_weights
        vsite_dict["x_weights"] = self._x_weights
        vsite_dict["y_weights"] = self._y_weights
        return vsite_dict

    @classmethod
    def from_dict(cls, vsite_dict):
        base_dict = deepcopy(vsite_dict)
        assert vsite_dict["vsite_type"] == "LocalVirtualSite"
        del base_dict["vsite_type"]
        return cls(**base_dict)

    @property
    def local_frame_weights(self):
        return self._o_weights, self._x_weights, self._y_weights

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
        type = ParameterAttribute(default="local", converter=_allow_only(["local"]))
        x_local = ParameterAttribute(unit=unit.nanometers)
        y_local = ParameterAttribute(unit=unit.nanometers)
        z_local = ParameterAttribute(unit=unit.nanometers)
        o_weights = IndexedParameterAttribute(converter=float)
        x_weights = IndexedParameterAttribute(converter=float)
        y_weights = IndexedParameterAttribute(converter=float)
        charge = ParameterAttribute(unit=unit.elementary_charge)
        sigma = ParameterAttribute(unit=unit.nanometer)
        epsilon = ParameterAttribute(unit.kilojoule_per_mole)

        @property
        def parent_index(self) -> int:
            """The parent is always the first index in a qubekit vsite"""
            return 0

        def get_weights(self) -> Tuple[List[float], ...]:
            return self.o_weights, self.x_weights, self.y_weights

        def to_openmm_particle(
            self, particle_indices: Tuple[int, ...]
        ) -> openmm.LocalCoordinatesSite:
            """Create an openmm local coord site based on the predefined weights using in QUBEKit"""
            o_weights, x_weights, y_weights = self.get_weights()
            return openmm.LocalCoordinatesSite(
                particle_indices,
                o_weights,
                x_weights,
                y_weights,
                openmm.Vec3(self.x_local, self.y_local, self.z_local),
            )

        def to_openff_particle(self, orientations: List[Tuple[int, ...]]):
            o_weights, x_weights, y_weights = self.get_weights()
            values_dict = {
                "p1": self.x_local,
                "p2": self.y_local,
                "p3": self.z_local,
                "o_weights": o_weights,
                "x_weights": x_weights,
                "y_weights": y_weights,
            }
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
        QUBEKitHandler,
        vdWHandler,
    ]

    exclusion_policy = ParameterAttribute(default="parents")

    def create_force(self, system: openmm.System, topology: Topology, **kwargs):
        # as the normal vsites go first there should be more than topology.n_atoms if we have a 4 site water or more
        if system.getNumParticles() < topology.n_topology_atoms:
            raise ValueError("the system does not seem to have enough particles in it")

        force = _get_nonbonded_force(system=system, topology=topology)

        matches_by_parent = self._find_matches_by_parent(topology)
        # order the matches by the index of the parent atom
        ordered_matches = sorted(list(matches_by_parent.keys()))

        parameter: LocalCoordinateVirtualSiteHandler.VirtualSiteType

        for parent_index in ordered_matches:
            parameters = matches_by_parent[parent_index]
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

    def _find_matches(
        self,
        entity: Topology,
        transformed_dict_cls=dict,
        unique=False,
    ) -> Dict[Tuple[int], List[ParameterHandler._Match]]:
        assigned_matches_by_parent = self._find_matches_by_parent(entity)
        return_dict = {}
        for parent_index, assigned_parameters in assigned_matches_by_parent.items():
            assigned_matches = []
            for assigned_parameter, match_orientations in assigned_parameters:
                for match in match_orientations:
                    assigned_matches.append(
                        ParameterHandler._Match(assigned_parameter, match)
                    )
            return_dict[(parent_index,)] = assigned_matches

        return return_dict


class UreyBradleyHandler(ParameterHandler):
    """
    Handle UreyBradley special angle class.

    This adds a harmonic angle term to the three tagged atoms and a harmonic bond to the terminal angle atoms.
    """

    class UBAngleType(ParameterType):
        """
        Smirnoff UB Angle type.
        """

        _VALENCE_TYPE = "Angle"
        _ELEMENT_NAME = "Angle"

        angle = ParameterAttribute(unit=unit.degree)
        angle_k = ParameterAttribute(unit=unit.kilojoule_per_mole / unit.degree**2)
        bond_length = ParameterAttribute(unit=unit.angstroms)
        bond_k = ParameterAttribute(unit=unit.kilocalorie_per_mole / unit.angstrom**2)

    _TAGNAME = "UreyBradley"
    _INFOTYPE = UBAngleType
    _OPENMMTYPE = openmm.HarmonicAngleForce
    _DEPENDENCIES = [ConstraintHandler, BondHandler, AngleHandler]

    def create_force(self, system: openmm.System, topology: Topology, **kwargs):
        """
        Add harmonic angles and bonds to the correct atoms in the system.
        """

        all_forces = {force.__class__.__name__: force for force in system.getForces()}
        angle_force = all_forces.get("HarmonicAngleForce", None)
        if angle_force is None:
            angle_force = self._OPENMMTYPE()
            system.addForce(angle_force)
        bond_force = all_forces.get("HarmonicBondForce", None)
        if bond_force is None:
            bond_force = openmm.HarmonicBondForce()
            system.addForce(bond_force)

        angle_matches = self.find_matches(topology, unique=False)
        skipped_constrained_angles = (
            0  # keep track of how many angles were constrained (and hence skipped)
        )
        for atoms, angle_match in angle_matches.items():
            try:
                self._assert_correct_connectivity(angle_match)
            except NotBondedError as e:
                smirks = angle_match.parameter_type.smirks
                raise NotBondedError(
                    f"While processing angle with SMIRKS {smirks}: " + e.msg
                )

            if (
                topology.is_constrained(atoms[0], atoms[1])
                and topology.is_constrained(atoms[1], atoms[2])
                and topology.is_constrained(atoms[0], atoms[2])
            ):
                # Angle is constrained; we don't need to add an angle term.
                skipped_constrained_angles += 1
                continue

            angle = angle_match.parameter_type
            angle_force.addAngle(*atoms, angle.angle, angle.angle_k)

            bond_force.addBond(atoms[0], atoms[-1], angle.bond_length, angle.bond_k)


class ProperRyckhaertBellemansHandler(ParameterHandler):
    """
    Handle RyckhaertBellemans torsion terms.
    """

    class ProperRBTorsionType(ParameterType):
        """
        A SMIRNOFF R-B torsion type.
        """

        _VALENCE_TYPE = "ProperTorsion"
        _ELEMENT_NAME = "Proper"

        c0 = ParameterAttribute(unit=unit.kilojoule_per_mole)
        c1 = ParameterAttribute(unit=unit.kilojoule_per_mole)
        c2 = ParameterAttribute(unit=unit.kilojoule_per_mole)
        c3 = ParameterAttribute(unit=unit.kilojoule_per_mole)
        c4 = ParameterAttribute(unit=unit.kilojoule_per_mole)
        c5 = ParameterAttribute(unit=unit.kilojoule_per_mole)

    _TAGNAME = "ProperRyckhaertBellemans"
    _INFOTYPE = ProperRBTorsionType
    _OPENMMTYPE = openmm.RBTorsionForce

    def create_force(self, system: openmm.System, topology: Topology, **kwargs):
        """
        Create the RBTorsion force and assign the torsions.
        """
        existing = [system.getForce(i) for i in range(system.getNumForces())]
        existing = [f for f in existing if type(f) == self._OPENMMTYPE]

        if len(existing) == 0:
            force = self._OPENMMTYPE()
            system.addForce(force)
        else:
            force = existing[0]

        torsion_matches = self.find_matches(topology)

        for atom_indices, torsion_match in torsion_matches.items():
            # Ensure atoms are actually bonded correct pattern in Topology
            try:
                self._assert_correct_connectivity(torsion_match)
            except NotBondedError as e:
                smirks = torsion_match.parameter_type.smirks
                raise NotBondedError(
                    f"While processing torsion with SMIRKS {smirks}: " + e.msg
                )
            parameter = torsion_match.parameter_type
            torsion_params = [
                parameter.c0,
                parameter.c1,
                parameter.c2,
                parameter.c3,
                parameter.c4,
                parameter.c5,
            ]
            force.addTorsion(*atom_indices, *torsion_params)


class BondChargeCorrectionHandler(ParameterHandler):
    """
    BondChargeCorrection handler to fit BCCs on top of AIM charges not to be used with Virtual sites?

    The charge correction is added to the first tagged atom and subtracted from the second, so swapping the indices is
    the same as swapping the sign of the correction.
    """

    class BCCType(ParameterType):
        """
        A SMIRNOFF style BCC type which adjusts Library charges with QUBEKit
        """

        _VALENCE_TYPE = "Bond"
        _ELEMENT_NAME = "BCC"

        charge_correction = ParameterAttribute(unit=unit.elementary_charge)

    _TAGNAME = "BondChargeCorrection"
    _INFOTYPE = BCCType
    _OPENMMTYPE = openmm.NonbondedForce

    _DEPENDENCIES = [
        vdWHandler,
        ElectrostaticsHandler,
        LibraryChargeHandler,
    ]  # we might need to depend on vdW if present

    def create_force(self, system, topology, **kwargs):
        """
        Apply the bond charge corrections to the system, charges must already be present
        """
        force = _get_nonbonded_force(system=system, topology=topology)

        bond_charge_matches = self.find_matches(topology)

        for match in bond_charge_matches.values():
            # the atom indices in the dict have been sorted for deduplication so use the ones in the match object
            atom_indices = match.environment_match.topology_atom_indices
            # Add charge to the first atom and subtracted from the second
            charge_added_atom, charge_subtracted_atom = atom_indices
            charge, sigma, epsilon = force.getParticleParameters(charge_added_atom)
            force.setParticleParameters(
                index=charge_added_atom,
                charge=charge + match.parameter_type.charge_correction,
                sigma=sigma,
                epsilon=epsilon,
            )

            charge, sigma, epsilon = force.getParticleParameters(charge_subtracted_atom)
            force.setParticleParameters(
                index=charge_subtracted_atom,
                charge=charge - match.parameter_type.charge_correction,
                sigma=sigma,
                epsilon=epsilon,
            )
