from pydantic import dataclasses


@dataclasses.dataclass
class LJData:
    a_i: float
    b_i: float
    r_aim: float


@dataclasses.dataclass
class FreeParams:
    # Beware weird units, (wrong in the paper too).
    # Units: vfree: Bohr ** 3, bfree: Ha * (Bohr ** 6), rfree: Angs
    v_free: float
    b_free: float
    r_free: float
