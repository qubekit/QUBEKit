# TODO Are these relative imports OK on travis?

# Import all of the core engines (shouldn't cause issues)
from .parameter_engines import XML, XMLProtein, AnteChamber

# try to import the extras
try:
    from .openforcefield import OpenFF
except ImportError:
    print('Openforcefield not available, continuing without for now. '
          'If you do not want to use it, please make sure it is removed from the config options; '
          'otherwise, install it. You can change parametrisation options with "-param <option>"')
    from .parameter_engines import Default as OpenFF
