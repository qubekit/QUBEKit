#!/usr/bin/env python3


def missing_import(name, fail_msg=''):
    """
    Generates a class which raises an import error when initialised.
    e.g. SomeClass = missing_import('SomeClass') will make SomeClass() raise ImportError
    """
    def init(self, *args, **kwargs):
        raise ImportError(
            f'The class {name} you tried to call is not importable; '
            f'this is likely due to it not doing installed. '
            f'{f"Fail Message: {fail_msg}" if fail_msg else ""}'
        )
    return type(name, (), {'__init__': init})


# try to import the extras
try:
    from .openforcefield import OpenFF
except ImportError:
    OpenFF = missing_import('OpenFF')

try:
    from .antechamber import AnteChamber
except ImportError:
    AnteChamber = missing_import('Antechamber')

try:
    from .xml import XML
except ImportError:
    XML = missing_import('XML')

try:
    from .xml_protein import XMLProtein
except ImportError:
    XMLProtein = missing_import('XMLProtein')
