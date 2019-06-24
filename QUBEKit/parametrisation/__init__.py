#!/usr/bin/env python3

# Import all of the core engines (shouldn't cause issues)
from .antechamber import AnteChamber
from .xml import XML
from .xml_protein import XMLProtein

# try to import the extras
try:
    from .openforcefield import OpenFF
except ImportError:
    print('Openforcefield not available, continuing without for now. '
          'If you do not want to use it, please make sure it is removed from the config options; '
          'otherwise, install it. You can change parametrisation options with "-param <option>"')
    from .base_parametrisation import Default as OpenFF
    setattr(OpenFF, 'name', 'OpenFF')
