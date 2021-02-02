#!/usr/bin/env python3

from QUBEKit.utils.helpers import try_load


AnteChamber = try_load('AnteChamber', 'QUBEKit.parametrisation.antechamber')
OpenFF = try_load('OpenFF', 'QUBEKit.parametrisation.openff')
Parametrisation = try_load('Parametrisation', 'QUBEKit.parametrisation.base_parametrisation')
XML = try_load('XML', 'QUBEKit.parametrisation.xml')
XMLProtein = try_load('XMLProtein', 'QUBEKit.parametrisation.xml_protein')
Gromacs = try_load("Gromacs", "QUBEKit.parametrisation.gromacs")
