#!/usr/bin/env python3

from QUBEKit.utils.exceptions import try_load


OpenFF = try_load('OpenFF', 'QUBEKit.parametrisation.openff')
AnteChamber = try_load('AnteChamber', 'QUBEKit.parametrisation.antechamber')
XML = try_load('XML', 'QUBEKit.parametrisation.xml')
XMLProtein = try_load('XMLProtein', 'QUBEKit.parametrisation.xml_protein')
