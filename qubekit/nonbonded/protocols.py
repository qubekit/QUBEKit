from typing_extensions import Literal

from qubekit.nonbonded.lennard_jones import LennardJones612
from qubekit.nonbonded.utils import FreeParams


# define the constants of the model for each element
#              v_free, b_free, r_free
def h_base(r_free: float):
    return FreeParams(7.6, 6.5, r_free)  # Polar Hydrogen symbol is X


def b_base(r_free: float):
    return FreeParams(46.7, 99.5, r_free)


def c_base(r_free: float):
    return FreeParams(34.4, 46.6, r_free)


def n_base(r_free: float):
    return FreeParams(25.9, 24.2, r_free)


def o_base(r_free: float):
    return FreeParams(22.1, 15.6, r_free)


def f_base(r_free: float):
    return FreeParams(18.2, 9.5, r_free)


def p_base(r_free: float):
    return FreeParams(84.6, 185, r_free)


def s_base(r_free: float):
    return FreeParams(75.2, 134.0, r_free)


def cl_base(r_free: float):
    return FreeParams(65.1, 94.6, r_free)


def br_base(r_free: float):
    return FreeParams(95.7, 162.0, r_free)


def si_base(r_free: float):
    return FreeParams(101.64, 305, r_free)


def i_base(r_free: float):
    return FreeParams(153.8, 385.0, r_free)


# Protocols as described in
# Exploration and Validation of Force Field Design Protocols through QM-to-MM Mapping

protocol_0 = {
    "free_parameters": {
        "H": h_base(r_free=1.738),
        "X": h_base(r_free=1.083),
        "C": c_base(r_free=2.008),
        "N": n_base(r_free=1.765),
        "O": o_base(r_free=1.499),
    },
}

protocol_1a = {
    "free_parameters": {
        "H": h_base(r_free=1.752),
        "X": h_base(r_free=1.111),
        "C": c_base(r_free=1.999),
        "N": n_base(r_free=1.740),
        "O": o_base(r_free=1.489),
    },
}

protocol_1b = {
    "free_parameters": {
        "H": h_base(r_free=1.737),
        "X": h_base(r_free=1.218),
        "C": c_base(r_free=2.042),
        "N": n_base(r_free=1.676),
        "O": o_base(r_free=1.501),
    },
}

protocol_2a = {
    "free_parameters": {
        "H": h_base(r_free=1.744),
        "X": h_base(r_free=1.107),
        "C": c_base(r_free=2.004),
        "N": n_base(r_free=1.708),
        "O": o_base(r_free=1.464),
    },
}

protocol_2b = {
    "free_parameters": {
        "H": h_base(r_free=1.724),
        "X": h_base(r_free=1.119),
        "C": c_base(r_free=2.026),
        "N": n_base(r_free=1.751),
        "O": o_base(r_free=1.521),
    },
}

protocol_2c = {
    "free_parameters": {
        "H": h_base(r_free=1.733),
        "X": h_base(r_free=1.132),
        "C": c_base(r_free=2.025),
        "N": n_base(r_free=1.756),
        "O": o_base(r_free=1.503),
    },
}

protocol_3a = {
    "free_parameters": {
        "H": h_base(r_free=1.670),
        "X": h_base(r_free=1.126),
        "C": c_base(r_free=2.051),
        "N": n_base(r_free=1.740),
        "O": o_base(r_free=1.590),
    },
}

protocol_3b = {
    "free_parameters": {
        "H": h_base(r_free=1.753),
        "X": h_base(r_free=1.404),
        "C": c_base(r_free=2.068),
        "N": n_base(r_free=1.681),
        "O": o_base(r_free=1.599),
    },
}

protocol_4a = {
    "free_parameters": {
        "H": h_base(r_free=1.719),
        "C": c_base(r_free=2.021),
        "N": n_base(r_free=1.604),
        "O": o_base(r_free=1.550),
    },
    "lj_on_polar_h": False,
}

protocol_4b = {
    "free_parameters": {
        "H": h_base(r_free=1.760),
        "X": h_base(r_free=1.154),
        "C": c_base(r_free=2.074),
        "N": n_base(r_free=1.742),
        "O": o_base(r_free=1.481),
    },
    "alpha": 1.301,
    "beta": 0.465,
}

protocol_5a = {
    "free_parameters": {
        "H": h_base(r_free=1.738),
        "X": h_base(r_free=1.279),
        "C": c_base(r_free=1.994),
        "N": n_base(r_free=1.706),
        "O": o_base(r_free=1.558),
    },
}

protocol_5b = {
    "free_parameters": {
        "H": h_base(r_free=1.747),
        "X": h_base(r_free=1.332),
        "C": c_base(r_free=2.019),
        "N": n_base(r_free=1.733),
        "O": o_base(r_free=1.598),
        "F": f_base(r_free=1.652),
        "Cl": cl_base(r_free=1.867),
        "Br": br_base(r_free=2.001),
        "S": s_base(r_free=2.036),
    },
    "alpha": 1.182,
    "beta": 0.445,
}

protocol_5c = {
    "free_parameters": {
        "H": h_base(r_free=1.687),
        "X": h_base(r_free=1.274),
        "C": c_base(r_free=2.042),
        "N": n_base(r_free=1.740),
        "O": o_base(r_free=1.630),
    },
}

protocol_5d = {
    "free_parameters": {
        "H": h_base(r_free=1.749),
        "X": h_base(r_free=1.447),
        "C": c_base(r_free=1.984),
        "N": n_base(r_free=1.717),
        "O": o_base(r_free=1.580),
        "F": f_base(r_free=1.628),
        "Cl": cl_base(r_free=1.831),
        "Br": br_base(r_free=1.964),
        "S": s_base(r_free=1.983),
    },
}

protocol_5e = {
    "free_parameters": {
        "H": h_base(r_free=1.680),
        "X": h_base(r_free=1.464),
        "C": c_base(r_free=2.043),
        "N": n_base(r_free=1.693),
        "O": o_base(r_free=1.680),
    },
    "alpha": 0.999,
    "beta": 0.491,
}

protocols = {
    "0": protocol_0,
    "1a": protocol_1a,
    "1b": protocol_1b,
    "2a": protocol_2a,
    "2b": protocol_2b,
    "2c": protocol_2c,
    "3a": protocol_3a,
    "3b": protocol_3b,
    "4a": protocol_4a,
    "4b": protocol_4b,
    "5a": protocol_5a,
    "5b": protocol_5b,
    "5c": protocol_5c,
    "5d": protocol_5d,
    "5e": protocol_5e,
}

MODELS = Literal[
    "0",
    "1a",
    "1b",
    "2a",
    "2b",
    "2c",
    "3a",
    "3b",
    "4a",
    "4b",
    "5a",
    "5b",
    "5c",
    "5d",
    "5e",
]


def get_protocol(protocol_name: MODELS) -> LennardJones612:
    """
    Get the LJ class configured for the desired protocol.
    """
    return LennardJones612.parse_obj(protocols[protocol_name])
