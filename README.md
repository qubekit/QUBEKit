# QUBEKit: QUantum BEspoke force field toolKit

**Newcastle University Cole Group.**


[QUBEKit](https://blogs.ncl.ac.uk/danielcole/qube-force-field/) is a python based force field derivation toolkit. It aims to allow users to quickly derive molecular mechanics parameters directly from quantum mechanical calculations. QUBEKit pulls together multiple pre-existing engines, as well as bespoke methods to produce accurate results with minimal user input.

Users who have used QUBEKit to derive any new force field parameters should cite the following papers:
* [Biomolecular Force Field Parameterization via Atoms-in-Molecule Electron Density Partitioning](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.6b00027)
* [Harmonic Force Constants for Molecular Mechanics Force Fields via Hessian Matrix Projection](https://pubs.acs.org/doi/10.1021/acs.jctc.7b00785)

To install, it is recommended to clone the QUBEKit folder into a home directory and run the install script. This will append the script to the search path:

    git clone git@github.com:cole-group/QuBeKit.git
    cd QUBEKit
    ./QUBE_install.sh
    
After relaunching the terminal you should now be able to use QUBEKit.

### Requirements

* [Anaconda3](https://www.anaconda.com/download/)
* [Gaussian09](http://gaussian.com/)

### Other Python modules used

Many packages come installed with the above requirements, however there are some which require further installation. All packages must be installed through conda and use conda as their path.
(Standard library packages such as collections are not listed.)

* psi4
* rdkit
* numpy

### In Development

QUBEKit should currently be considered a work in progress.
While it is stable we are constantly working to improve the code and increase the amount of compatible software. 
Please also be aware that some bugs may not be our fault!
We use lots of software written by many different people;
if reporting a bug please make sure it is a bug with QUBEKit and not with a dependency.   

# Help

Below is general help for most of the commands available in QUBEKit. 

## QUBEKit Commands: Running Jobs

Given a list of commands, such as: "-bonds", "-h", "psi4" some are taken as single word commands.
Others however, such as changing defaults: ("-c", "0") ("-m", "1") etc are taken as tuple commands.
All tuple commands are preceded by a "-", while single word commands are not.
(An error is raised for hanging commands e.g. "-c", "1" or "-h".)

Single-word commands are taken in any order, assuming they are given, tuple commands are taken as ("-key", "val").
All commands are optional. If nothing at all is given, the program will run entirely with defaults.

Files to be analysed must be written with their file extension (.pdb) attached or they will not be recognised commands.
All commands should be given in lower case
(you may use whatever case you like for the name of files (e.g. DMSO.pdb) or log strings (e.g. Run013).

### Some Examples

Running a charges analysis with a non-default charge of 1 and the default charge engine (chargemol):
Note, ordering does not matter as long as tuples commands are together.
    
    python run.py molecule.pdb charges -c 1
    python run.py -c 1 molecule.pdb charges

Running a bonds analysis with a non-default engine (g09):

    python run.py molecule.pdb -bonds g09

The program will tell the user which defaults are being used, and which commands were given.
Errors will be raised for any invalid commands and the program will not run.

Each time the program runs, a new working directory containing a log file will be created.
The name of the directory will contain the run number provided via the terminal command or the default run number in the config if none is provided).
This log file will store which methods were called, how long they took, and any docstring for them (if it exists).
The log file will also contain information regarding the config options used, as well as the commands given.

The format for the name of the active directory is:

    QUBEKit_moleculename_YYYY_MM_DD_runnumber
    
The format for the log file is similar:

    QUBEKit_log_moleculename_YYYY_MM_DD_runnumber

If using QUBEKit multiple times per day with the same molecule, it is therefore necessary to update the 'run number'.
Not updating the run number when analysing the same molecule on the same day will prevent the program from running.
Updating the run number can be done with the command:
python run.py -log #
where '#' is any number or string you like (no spaces).

### High Throughput

Bulk commands are for high throughput analysis; they are invoked with the -bulk keyword.
A csv must be used when running a bulk analysis. This csv contains the defaults for each molecule being analysed.
Before running a bulk analysis, fill in each column for each molecule.

If running a bulk analysis with smiles strings, use the command:

    python run.py -bulk smiles example.csv
    
where example.csv is the csv file containing all the information described above.

If running a bulk analysis with pdb files, use the command:

    python run.py -bulk pdb example.csv

again, example.csv is the csv containing the necessary defaults for each molecule.
The pdb files should all be in the same place: where you're running QUBEKit from.

Please note, there are deliberately two config files.
The changeable parameters are spread across a .csv and a .py config files.
The configs in the .py are more likely to be kept constant across a bulk analysis.
For this reason, the .csv config contains highly specific parameters such as torsions which will change molecule to molecule.
The .py however contains more typically static parameters such as the basis sets and engines being used (e.g. psi4, chargemol, etc).
If you would like the .py config to change from molecule to molecule, you may specify that in the .csv config.

You may change defaults inside the terminal when running bulk analyses, and these changed defaults will be printed to the log file.
However, the config files themselves will not be overwritten.
It is therefore recommended to manually edit the config files rather than doing, for example:

    python run.py -bulk pdb example.csv -log run42 -ddec 3 -solvent true
    
Be aware that the names of the pdb files are used as keys to find the configs.
So, each pdb being analysed should have a corresponding row in the csv file with the correct name.

For example (csv row order does not matter, and you do not need to include smiles strings when a pdb is provided):

    PATH/files/:
        benzene.pdb
        ethane.pdb
        methane.pdb

    config.csv:
    name,charge,multiplicity,config,smiles string,torsion order
    default,0,1,default_config,,
    methane,0,1,default_config,C,
    benzene,0,1,default_config,,
    ethane,0,1,default_config,,

If you would like to generate a blank csv config file, simply run the command:

    python run.py -csv example.csv
    
where example.csv is the name of the config file you want to create.
This will automatically generate the file with the appropriate columns and a row with defaults inside.
The csv config file will be put into the configs folder.
When writing to the csv file, append rows after the defaults row rather than overwriting it.

### Notes on Commands

You cannot run multiple kinds of analysis at once. For example:

    python run.py -bulk pdb example.csv methane.pdb -bonds g09
    
is not a valid command. These should be performed separately:

    python run.py -bulk pdb example.csv
    python run.py methane.pdb -bonds g09