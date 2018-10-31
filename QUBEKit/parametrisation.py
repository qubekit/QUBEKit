# TODO write the parametrisation classes for each method antechamber input xml, openFF, etc
# all must return the same dic object that can be stored in the molecule and writen to xml format
# maybe gromacs as well


class Parametrisation:
    """Class of functions which perform the initial parametrisation for the molecules."""

    def __init__(self, molecule, config_file):
        self.molecule = molecule



class antechamber(Parametrisation):
        pass

class openff(Parametrisation):
        pass


class BOSS(Parametrisation):
        pass
