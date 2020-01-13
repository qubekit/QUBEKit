#!/usr/bin/env python3

from collections import namedtuple

KCAL_TO_KJ = 4.184
KJ_TO_KCAL = 0.23900573613

HA_TO_KCAL_P_MOL = 627.509391
KCAL_P_MOL_TO_HA = 0.00159360164

NM_TO_ANGS = 10
ANGS_TO_NM = 0.1

BOHR_TO_ANGS = 0.529177
ANGS_TO_BOHR = 1.88972687777

STP = 298.15

PI = 3.141592653589793
DEG_TO_RAD = PI / 180
RAD_TO_DEG = 180 / PI

# Boltzmann constant in Kcal/(mol * K)
KB_KCAL_P_MOL_K = 0.0019872041


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
