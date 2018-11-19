# TODO write the parametrisation classes for each method antechamber input xml, openFF, etc
# all must return the same dic object that can be stored in the molecule and writen to xml format
# maybe gromacs as well


from helpers import config_loader


class Parametrisation:
    """Class of functions which perform the initial parametrisation for the molecules."""

    def __init__(self, molecule, config_file):

        self.engine_mol = molecule
        confs = config_loader(config_file)
        self.qm, self.fitting, self.paths = confs


class Antechamber(Parametrisation):

    def __init__(self, molecule, config_file):

        super().__init__(molecule, config_file)


class OpenFF(Parametrisation):

    def __init__(self, molecule, config_file):

        super().__init__(molecule, config_file)


class BOSS(Parametrisation):

    def __init__(self, molecule, config_file):

        super().__init__(molecule, config_file)
