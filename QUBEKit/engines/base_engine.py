#!/usr/bin/env python3


class Engines:
    """
    Engines superclass containing core information that all other engines (PSI4, Gaussian etc) will have.
    Provides atoms' coordinates with name tags for each atom and entire molecule.
    Also gives all configs from the appropriate config file.
    """

    def __init__(self, molecule):

        self.molecule = molecule

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__!r})'


class Default:
    """
    If there is an import error, this class replaces the class which failed to be imported.
    Then, only if initialised, an import error will be raised notifying the user of a failed call.
    """

    def __init__(self, *args, **kwargs):
        # self.name is set when the failed-to-import class is set to Default.
        raise ImportError(
            f'The class {self.name} you tried to call is not importable; '
            f'this is likely due to it not being installed.')

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__!r})'
