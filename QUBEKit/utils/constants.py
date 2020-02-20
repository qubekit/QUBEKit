#!/usr/bin/env python3

"""
All conversions are multiplicative, e.g.
NM_TO_ANGS; 6 nm * NM_TO_ANGS = 60 Angs
Division will give the wrong conversion.
"""

from collections import namedtuple


AVOGADRO = 6.02214179e23        # Particles in 1 Mole
ROOM_TEMP = 298.15              # Kelvin
ROOM_PRESSURE = 101325          # Pascals
VACUUM_PERMITTIVITY = 8.8541878128e-12      # Farads per Metre
ELECTRON_CHARGE = 1.602176634e-19           # Coulombs
KB_KCAL_P_MOL_K = 0.0019872041      # Boltzmann constant in KCal/(mol * K)

PI = 3.141592653589793
DEG_TO_RAD = PI / 180
RAD_TO_DEG = 180 / PI


KCAL_TO_KJ = 4.184
KJ_TO_KCAL = 0.23900573613
J_TO_KCAL = 0.0002390057

J_TO_KCAL_P_MOL = J_TO_KCAL * AVOGADRO

HA_TO_KCAL_P_MOL = 627.509391
KCAL_P_MOL_TO_HA = 0.00159360164

NM_TO_ANGS = 10
ANGS_TO_NM = 0.1

ANGS_TO_M = 10e-10

BOHR_TO_ANGS = 0.529177
ANGS_TO_BOHR = 1.88972687777


# Used for printing colours to terminal. Wrap a colour and end around a block like so:
# f'{COLOURS.red}sample message here{COLOURS.end}'
Colours = namedtuple('colours', 'red green orange blue purple end')

# Uses exit codes to set terminal font colours.
# \033[ is the exit code. 1;32m are the style (bold); colour (green) m reenters the code block.
# The end code resets the style back to default.
COLOURS = Colours(
    red='\033[1;31m',
    green='\033[1;32m',
    orange='\033[1;33m',
    blue='\033[1;34m',
    purple='\033[1;35m',
    end='\033[0m'
)
