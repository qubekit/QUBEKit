from typing import ClassVar, Dict

from pydantic import Field, validator
from typing_extensions import Literal

from qubekit.charges.solvent_settings.base import SolventBase
from qubekit.utils import constants
from qubekit.utils.exceptions import SpecificationError


class SolventPsi4(SolventBase):

    program: Literal["psi4"] = "psi4"
    units: Literal["au", "angstrom"] = Field(
        ...,
        description="The units used in the input options atomic units are used by default.",
    )
    codata: Literal[1998, 2002, 2006, 2010] = Field(
        2010,
        description="The set of fundamental physical constants to be used in the module.",
    )
    cavity_Type: Literal["GePol"] = Field(
        "GePol",
        description="Completely specifies type of molecular surface and its discretization.",
    )
    cavity_Area: float = Field(
        0.3,
        description="Average area (weight) of the surface partition for the GePol cavity in the specified units. By default this is in AU.",
    )
    cavity_Scaling: bool = Field(
        True,
        description="If true, the radii for the spheres will be scaled by 1.2. For finer control on the scaling factor for each sphere, select explicit creation mode.",
    )
    cavity_RadiiSet: Literal["bondi", "uff", "allinger"] = Field(
        "Bondi",
        description="Select set of atomic radii to be used. Currently Bondi-Mantina Bondi, UFF  and Allinger’s MM3 sets available. Radii in Allinger’s MM3 set are obtained by dividing the value in the original paper by 1.2, as done in the ADF COSMO implementation We advise to turn off scaling of the radii by 1.2 when using this set.",
    )
    cavity_MinRadius: float = Field(
        100,
        description="Minimal radius for additional spheres not centered on atoms. An arbitrarily big value is equivalent to switching off the use of added spheres, which is the default in AU.",
    )
    cavity_Mode: Literal["Implicit"] = Field(
        "Implicit",
        description="How to create the list of spheres for the generation of the molecular surface.",
    )
    medium_SolverType: Literal["IEFPCM", "CPCM"] = Field(
        "IEFPCM",
        description="Type of solver to be used. All solvers are based on the Integral Equation Formulation of the Polarizable Continuum Model.",
    )
    medium_Nonequilibrium: bool = Field(
        False,
        description="Initializes an additional solver using the dynamic permittivity. To be used in response calculations.",
    )
    medium_Solvent: str = Field(
        ...,
        description="Specification of the dielectric medium outside the cavity. Note this will always be converted to the molecular formula to aid parsing via PCM.",
    )
    medium_MatrixSymm: bool = Field(
        True,
        description="If True, the PCM matrix obtained by the IEFPCM collocation solver is symmetrized.",
    )
    medium_Correction: float = Field(
        0.0,
        description="Correction, k for the apparent surface charge scaling factor in the CPCM solver.",
        ge=0,
    )
    medium_DiagonalScaling: float = Field(
        1.07,
        description="Scaling factor for diagonal of collocation matrices, values commonly used in the literature are 1.07 and 1.0694.",
        ge=0,
    )
    medium_ProbeRadius: float = Field(
        1.0,
        description="Radius of the spherical probe approximating a solvent molecule. Used for generating the solvent-excluded surface (SES) or an approximation of it. Overridden by the built-in value for the chosen solvent. Default in AU.",
    )
    _solvents: ClassVar[Dict[str, str]] = {
        "water": "H2O",
        "dimethylsulfoxide": "DMSO",
        "nitromethane": "CH3NO2",
        "acetonitrile": "CH3CN",
        "methanol": "CH3OH",
        "ethanol": "CH3CH2OH",
        "1,2-dichloroethane": "C2H4CL2",
        "methylenechloride": "CH2CL2",
        "tetrahydrofurane": "THF",
        "aniline": "C6H5NH2",
        "chlorobenzene": "C6H5CL",
        "chloroform": "CHCL3",
        "toluene": "C6H5CH3",
        "1,4-dioxane": "C4H8O2",
        "carbon tetrachloride": "CCL4",
        "cyclohexane": "C6H12",
        "n-heptane": "C7H16",
    }

    @validator("medium_Solvent")
    def _check_solvent(cls, solvent: str) -> str:
        """
        Make sure that a valid solvent from the list of supported values is passed.
        """

        solvent_formula = cls._solvents.get(solvent.lower(), solvent.upper())
        if solvent_formula not in cls._solvents.values():
            raise SpecificationError(
                f"The solvent {solvent} is not supported please chose from the following solvents or formulas {cls._solvents.items()}"
            )
        return solvent_formula

    def __init__(self, **kwargs):
        """
        Fully validate the model making sure options are compatible and convert any defaults to the give unit system.
        """
        # convert all inputs to the correct units
        units = kwargs.get("units", None)
        if units is not None and units.lower() == "angstrom":
            # we need to convert the default values only which have length scales
            if "medium_ProbeRadius" not in kwargs:
                medium_ProbeRadius = (
                    self.__fields__["medium_ProbeRadius"].default
                    * constants.BOHR_TO_ANGS
                )
                kwargs["medium_ProbeRadius"] = medium_ProbeRadius
            if "cavity_MinRadius" not in kwargs:
                cavity_MinRadius = (
                    self.__fields__["cavity_MinRadius"].default * constants.BOHR_TO_ANGS
                )
                kwargs["cavity_MinRadius"] = cavity_MinRadius
            if "cavity_Area" not in kwargs:
                cavity_Area = (
                    self.__fields__["cavity_Area"].default * constants.BOHR_TO_ANGS ** 2
                )
                kwargs["cavity_Area"] = cavity_Area
        super(SolventPsi4, self).__init__(**kwargs)

    def format_keywords(self) -> str:
        """
        Generate the formatted PCM settings string which can be ingested by psi4 via the qcschema interface.
        """
        # format the medium keywords
        medium_str, cavity_str = "", ""
        for prop in self.__fields__.keys():
            if "medium" in prop:
                medium_str += f"\n     {prop[7:]} = {getattr(self, prop)}"
            elif "cavity" in prop:
                cavity_str += f"\n     {prop[7:]} = {getattr(self, prop)}"
        # format the cavity keywords
        pcm_string = f"""
        Units = {self.units}
        CODATA = {self.codata}
        Medium {{{medium_str
        }}}
        Cavity {{{cavity_str}}}"""
        return pcm_string
