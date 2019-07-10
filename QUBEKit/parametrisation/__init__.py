#!/usr/bin/env python3

from QUBEKit.exceptions import try_load


OpenFF = try_load('OpenFF', 'QUBEKit.parametrisation.openforcefield')
AnteChamber = try_load('AnteChamber', 'QUBEKit.parametrisation.antechamber')
XML = try_load('XML', 'QUBEKit.parametrisation.xml')
XMLProtein = try_load('XMLProtein', 'QUBEKit.parametrisation.xml_protein')
