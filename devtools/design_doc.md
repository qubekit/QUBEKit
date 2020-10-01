The aim of this document is to outline a general overview of how QUBEKit works from input to result.

#### The Molecule Object

Throughout QUBEKit, we use a class which we call the "Molecule Object".
This class acts as a container which has all of the necessary information for an execution.
Configs, coordinates, home directory, etc is all stored in this object.
As a result, a lot of the main execution methods and classes will take a molecule object.
Whenever a class or method takes an argument: `molecule`, this is what is meant.

The molecule object is either a ligand or a protein, the code for which is stored in the `ligand.py` file.
A molecule object is initialised by simply passing a filename, smiles string or json dict like so:
    
    from QUBEKit.ligand import Ligand
    
    methane = Ligand('methane.pdb')
    ethane = Ligand('CC')
    ethane_with_name = Ligand(['CC', 'ethane'])
    propane = Ligand('propane.mol2')
    ...
    
Upon initialisation, QUBEKit automatically builds the topology for the object, infers charge, multiplicity
and calculates all of the bonds, angles, dihedrals etc within the molecule.
    
Once the information is stored into the molecule object, it can be easily accessed and mutated.

    >>> molecule = Ligand('methane.pdb')
    >>> molecule.charge
    0

Various files can also be produced from the molecule object:

    >>> molecule.write_pdb()
    >>> molecule.write_parameters() # produce an xml file
    
Changing the acceptable inputs can be fairly easily achieved by changing the available methods in `file_handlers.py`.
It is also easy to add extra information by adding more arguments to the molecule object. 

#### Engines

QUBEKit is split up into several "Engines" which are (usually) classes encompassing a particular external package such as Gaussian, PSI4 or RDKit.
These classes have methods for their respective jobs such as an RDKit file reader: `RDKit.read_file()`.
Engines are designed to be as general as possible, and have a reasonable number of defaults.
This allows them to be called with minimal user understanding of features they might not need.
    
    # Only requires a filename.
    RDKit.read_file(<filename>)
    # Only requires a molecule object;
    # a PSI4 run file then be written and executed based on the information provided.
    PSI4(<molecule>).generate_input()

#### run.py

`run.py` is the main execution file. It contains a class for handling input and configs `ArgsAndConfigs`,
and a class for execution `Execute`. 

`ArgsAndConfigs` does the following:

* Checks for a master config file
* Parses any terminal commands,
* Initialises a molecule object
* Saves any necessary changes into that ligand object
* Executes 

Adding commands can therefore be fairly easily achieved by just adding them to the `parse_commands()` method.

`Execute` contains methods for each of the key stages of a QUBEKit execution.
When running QUBEKit, each of these stages will produce a folder which contains all the relevant information for that stage.
Each stage takes a molecule object, uses its parameters to perform some calculation, then returns the mutated molecule object.

For example, `lennard_jones()` takes a molecule object which contains the charge information.
QUBEKit uses this in conjunction with some DDEC files to calculate the Lennard-Jones parameters and save them back into the molecule object.

Between each stage, the molecule object is saved (using a python module, "pickle") and some logging is performed.

This is achieved by wrapping each function in `Execute` with the method`stage_wrapper()`.
`stage_wrapper()` will start by loading the saved molecule object from the previous stage;
it will then generate a new folder with the appropriate name (e.g. 06_density etc.);
run the particular stage (e.g. `density()`), then re-save the newly mutated molecule object.

More succinctly, each stage undergoes the following process:

* *Load* molecule object
* *Print* "Starting stage ..." or similar
* *Make* and *move* into a new *folder* for that stage
* *Execute* that stage
* *Print* "Finished stage ..." or similar
* *Save* the molecule object

#### Logging and Saving State

Any time QUBEKit encounters an unhandlable exception at runtime, before QUBEKit quits, the exception is logged. 
The molecule object is also saved; its current state is printed to the log file;
and the exception is printed to the log file too. 

This helps identify the issue and prevent recurrence, as well as making it easier to restart from a successful point, rather than the beginning.

If the logger itself encounters an exception (very unlikely, never happened before), the exception will just be raised as normal.

"The molecule object is saved" means that the molecule object, which contains all of the information
for the particular execution, is stored in a file. This file is called `.QUBEKit_states`.
It is a hidden file which is generated using python's builtin module "pickle".
It allows the entire execution to be saved and loaded from a particular state using the QUBEKit functions:
`pickle()` and `unpickle()`.
The molecule object has the method pickle which will store all of the execution information to the pickle file `.QUBEKit_states`.
This information can then be accessed by using `utils.helpers.unpickle()`.  

#### Using QUBEKit as an Importable Module

QUBEKit is designed to be simple to use when imported, not just when executed from the terminal.
This is achieved by having a highly modular design.
Often, all that is needed is to initialise a molecule object, then call the relevant methods on that object.

Say, for example, a user would like to apply the modified Seminario method to a precalculated hessian matrix to gather the bond and angle lengths and force constants.
All that would be needed is some input file containing the coordinates, and the hessian matrix itself.

```python
from QUBEKit.ligand import Ligand
from QUBEKit.mod_seminario import ModSeminario

import numpy as np


# Initialise molecule
molecule = Ligand('molecule.pdb')
# Load hessian from file
hessian = np.load('hessian.npy')
# Insert hessian into molecule object
molecule.hessian = hessian
# Calculate force constants, angles and bond lengths
mod_sem = ModSeminario(molecule)
mod_sem.modified_seminario_method()
# (optional) Symmetrise based on topology
mod_sem.symmetrise_bonded_parameters()

# Return an updated pdb and xml
molecule.write_parameters()
```
