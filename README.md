# QUBEKit - *Qu*antum Mechanical *Be*spoke force field tool*kit*

#### **Newcastle University UK - Cole Group**

| **Status** | [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/qubekit/QUBEKit.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/qubekit/QUBEKit/context:python) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  [![CI](https://github.com/qubekit/QUBEKit/actions/workflows/CI.yaml/badge.svg)](https://github.com/qubekit/QUBEKit/actions/workflows/CI.yaml)  [![codecov](https://codecov.io/gh/qubekit/QUBEKit/branch/master/graph/badge.svg?token=E554IAATJC)](https://codecov.io/gh/qubekit/QUBEKit)|
| :------ | :------ |
| **Foundation** | [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![python](https://img.shields.io/badge/python-3.6%2C%203.7%2C%203.8-blue.svg)](https://www.python.org/) [![platforms](https://anaconda.org/conda-forge/qubekit/badges/platforms.svg)]() [![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/qubekit?color=blue&logo=anaconda&logoColor=white)](https://anaconda.org/conda-forge/qubekit)|
| **Installation** | [![Anaconda-Server Badge](https://anaconda.org/conda-forge/qubekit/badges/installer/conda.svg)](https://anaconda.org/conda-forge/qubekit) [![Anaconda-Server Badge](https://anaconda.org/conda-forge/qubekit/badges/downloads.svg)](https://anaconda.org/conda-forge/qubekit) [![Anaconda-Server Badge](https://anaconda.org/conda-forge/qubekit/badges/latest_release_date.svg)](https://anaconda.org/conda-forge/qubekit) |

## Table of Contents

* [What is QUBEKit?](https://github.com/qubekit/QUBEKit#what-is-qubekit)
    * [Development](https://github.com/qubekit/QUBEKit#in-development)
* [Installation](https://github.com/qubekit/QUBEKit#installation)
    * [Requirements](https://github.com/qubekit/QUBEKit#requirements)
* [Help](https://github.com/qubekit/QUBEKit#help)
    * [Config Files](https://github.com/qubekit/QUBEKit#before-you-start-config-files)
    * [QUBEKit Commands](https://github.com/qubekit/QUBEKit#qubekit-commands-running-jobs)
        * [Running Jobs](https://github.com/qubekit/QUBEKit#qubekit-commands-running-jobs)
        * [Some Examples](https://github.com/qubekit/QUBEKit#qubekit-commands-some-examples)
        * [Logging](https://github.com/qubekit/QUBEKit#qubekit-commands-some-examples)
        * [High Throughput](https://github.com/qubekit/QUBEKit#qubekit-commands-high-throughput)
        * [Custom Start and End Points](https://github.com/qubekit/QUBEKit#qubekit-commands-custom-start-and-end-points-single-molecule)
            * [Single Molecules](https://github.com/qubekit/QUBEKit#qubekit-commands-custom-start-and-end-points-single-molecule)
            * [Multiple Molecules](https://github.com/qubekit/QUBEKit#qubekit-commands-custom-start-and-end-points-multiple-molecules)
        * [Checking Progress](https://github.com/qubekit/QUBEKit#qubekit-commands-checking-progress)
        * [Other Commands and Information](https://github.com/qubekit/QUBEKit#qubekit-commands-other-commands-and-information)
* [Cook Book](https://github.com/qubekit/QUBEKit#cook-book)

## What is QUBEKit?

[QUBEKit](https://blogs.ncl.ac.uk/danielcole/qube-force-field/) is a Python 3.6+ based force field derivation toolkit for Linux operating systems.
Our aims are to allow users to quickly derive molecular mechanics parameters directly from quantum mechanical calculations.
QUBEKit pulls together multiple pre-existing engines, as well as bespoke methods to produce accurate results with minimal user input.
QUBEKit aims to avoid fitting to experimental data where possible while also being highly customisable.

Users who have used QUBEKit to derive any new force field parameters should cite the following papers:

* [QUBEKit: Automating the Derivation of Force Field Parameters from Quantum Mechanics](https://pubs.acs.org/doi/full/10.1021/acs.jcim.8b00767)
* [Biomolecular Force Field Parameterization via Atoms-in-Molecule Electron Density Partitioning](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.6b00027)
* [Harmonic Force Constants for Molecular Mechanics Force Fields via Hessian Matrix Projection](https://pubs.acs.org/doi/10.1021/acs.jctc.7b00785)

### In Development

QUBEKit should currently be considered a work in progress.
While it is stable we are constantly working to improve the code and broaden its compatibilities. 

We use lots of software written by many different people;
if reporting a bug please (to the best of your ability) make sure it is a bug with QUBEKit and not with a dependency.
We welcome any suggestions for additions or changes.

## Installation

QUBEKit is now available through conda-forge; this is the recommended installation method.
Github has our latest version which will likely have newer features but may not be stable.
Pip is not updated as regularly as Anaconda or Github but should also be stable.

    # Recommended
    conda install -c conda-forge qubekit

---
    
    # Alternative 
    pip install qubekit

---
    
    # Recommended for Developers (see below)
    git clone https://github.com/qubekit/QUBEKit.git
    cd <install location>
    python setup.py install    

### Requirements

* [Anaconda3](https://www.anaconda.com/download/)

Download Anaconda from the above link and install with the linux command:

    ./Anaconda3<version>.sh

*You may need to use `chmod +x Anaconda3<version>.sh` to make it executable.*

We recommend you add conda to your .bashrc when prompted.

* [Gaussian09](http://gaussian.com/)

Installation of Gaussian is likely handled by your institution; QUBEKit uses it for density calculations only.
If you don't plan on performing these sorts of calculations then it is not necessary.
If you do, please make sure Gaussian09 is executable with the command `g09`.

* [Chargemol](https://sourceforge.net/projects/ddec/files/)

Chargemol can be downloaded and installed from a zip file in the above link. 
Be sure to add the path to the QUBEKit configs once you've generated them *([explanation](https://github.com/qubekit/QUBEKit#before-you-start-config-files))*.

Most conda packages are included in the conda-forge install.
Packages not available through conda-forge may need to be installed separately.

    conda install -c conda-forge qubekit

The following table details some of the core requirements included in the conda install of QUBEKit.
If any packages are missing from the install or causing issues, this table shows how to get them.

| **Package** | Conda Install |
| :------ | :------ |
| [GeomeTRIC](https://github.com/leeping/geomeTRIC) | `conda install -c conda-forge geometric` |
| [OpenForceField](https://openforcefield.org/) | `conda install -c conda-forge openff-toolkit` |
| [OpenMM](http://openmm.org/) | `conda install -c omnia openmm` |
| [PSI4](http://www.psicode.org/) | `conda install -c psi4 psi4` |
| [QCEngine](https://pypi.org/project/qcengine/) | `conda install -c conda-forge qcengine` |
| [RDKit](http://rdkit.org/) | `conda install -c rdkit rdkit` |
| [TorsionDrive](https://github.com/lpwgroup/torsiondrive) | `conda install -c conda-forge torsiondrive` |

Adding lots of packages can be a headache. If possible, install using Anaconda through the terminal.
This is generally safest, as Anaconda should deal with versions and conflicts in your environment.
Generally, conda packages will have the conda install command on their website or github.
For the software not available through Anaconda, or if Anaconda is having trouble resolving conflicts, either git clone them and install:

    git clone https://<git_address_here>
    cd <location of cloned package>
    python setup.py install

or follow the described steps in the respective documentation.

#### Installing as dev

If downloading QUBEKit to edit the latest version of the source code, 
the easiest method is to install via conda, then remove the conda version of qubekit and git clone.
This is accomplished with a few simple commands:
    
    # Install QUBEKit as normal
    conda install -c conda-forge qubekit
    
    # Remove ONLY the QUBEKit package itself, leaving all dependencies installed
    # and on the correct version
    conda remove --force qubekit
    
    # Re-download the latest QUBEKit through github
    git clone https://github.com/qubekit/QUBEKit.git
    
    # Re-install QUBEKit outside of conda
    cd QUBEKit
    python setup.py install

## Help

Below is general help for most of the commands available in QUBEKit.
There is some short help available through the terminal (invoked with `-h`) 
but all necessary long-form help is within this document.

### Config files

QUBEKit has a lot of settings which are used in production and changing these can result in very different force field parameters.
The settings are controlled using ini style config files which are easy to edit.
After installation you should notice a `QUBEKit_configs` 
folder in your main home directory; now you need to create a master template.
To do this, use the command `QUBEKit -setup` where you will be presented with the following:

    You can now edit config files using QUBEKit, chose an option to continue:
    1) Edit a config file
    2) Create a new master template
    3) Make a normal config file

Choose option two to set up a new template which will be used every time you run QUBEKit 
(unless you supply the name of another ini file in the configs folder).
The only parameter that ***must*** be changed for QUBEKit to run is the Chargemol path in the descriptions section.
This option is what controls where the Chargemol code is accessed from on your PC.
It should be the location of the Chargemol home directory, plus the name of the Chargemol folder itself to account for version differences:

    '/home/<user>/Programs/chargemol_09_26_2017'
    
Following this, feel free to change any of the other options such as the basis set.

QUBEKit ***does*** have a full suite of defaults built in. 
You do not necessarily need to create and manage an ini config file; everything can be done through the terminal commands.
To make it easier to keep track of changes however, we recommend you do use a config file, 
or several depending on the analysis you're doing.

You can change which config file is being used at runtime using the command:

    -config <config file name>.ini
    
Otherwise, the default `master_config.ini` will be used.

### QUBEKit Commands: Running Jobs

Running a job entirely on defaults, is as simple as typing `-i` for input, followed by the pdb file name, for example:

    QUBEKit -i methane.pdb
    
This will perform a start-to-finish analysis on the `methane.pdb` file using the default config ini file.
For anything more complex, you will need to add more commands.

---

Given a list of commands, such as: `-setup`, `-progress` some are taken as single word commands.
Others however, such as changing defaults: (`-c 0`), (`-m 1`), are taken as tuple commands.
The first command of tuple commands is always preceded by a `-`, while the latter commands are not: (`-skip` `density` `charges`).
(An error is raised for 'hanging' commands e.g. `-c`, `1` or `-sm`.)

All commands can be provided in any order, as long as tuple commands are paired together.
**All configuration commands are optional**. If nothing at all is given, the program will run entirely with defaults.
QUBEKit only *needs* to know the molecule you're analysing, given with `-i` `<molecule>.pdb` or `-sm` `<smiles string>`.

Files to be analysed must be written with their file extension (.pdb) attached or they will not be recognised commands.
All commands should be given in lower case with two main exceptions;
you may use whatever case you like for the name of files (e.g. `-i DMSO.pdb`) or the name of the directory (e.g. `-log Run013`).

### QUBEKit Commands: Some Examples

*A full list of the possible command line arguments is given below in the 
[Cook Book](https://github.com/qubekit/QUBEKit#cook-book) section. 
This section covers some simple examples*

Running a full analysis on `molecule.pdb` with a non-default charge of `1`, the default charge engine `Chargemol` and with GeomeTRIC `off`:
Note, ordering does not matter as long as tuples commands (`-c` `1`) are together.

`-i` is for the input, `-c` denotes the charge and `-geo` is for (en/dis)abling geomeTRIC.
    
    QUBEKit -i molecule.pdb -c 1 -geo false
    QUBEKit -c 1 -geo false -i molecule.pdb

Running a full analysis with a non-default bonds engine: Gaussian09 (g09):

    QUBEKit -i molecule.pdb -bonds g09

The program will tell the user which defaults are being used, and which commands were given.
Errors will be raised for any invalid commands and the program will not run.
A full log of what's happening will be created in a `QUBEKit_log.txt` file.

Try running QUBEKit with the command:

    QUBEKit -sm C methane -end hessian

This will generate a methane pdb file (and mol file) using its smiles string: `C`,
then QUBEKit will analyse it until the hessian is calculated.
See [QUBEKit Commands: Custom Start and End Points (single molecule)](https://github.com/qubekit/QUBEKit#qubekit-commands-custom-start-and-end-points-single-molecule) below for more details on `-end`.

### QUBEKit Commands: Logging

Each time QUBEKit runs, a new working directory containing a log file will be created.
The name of the directory will contain the run number or name provided via the terminal command `-log` 
(or the run number or name from the configs if a `-log` command is not provided).
The log file will contain information regarding the config options used, as well as the commands given and much more.
The log file updates in real time and contains more information than is printed to the terminal during a run.
If there is an error with QUBEKit, the full stack trace of an exception will be stored in the log file.
Many errors have custom exceptions to help elucidate if, for example, a module has not been installed correctly.

**The error printed to the terminal may be different, truncated or incorrect so it's always better to check the log file.**

The format for the name of the active directory is:

    QUBEKit_moleculename_YYYY_MM_DD_runnumber(_backupnumber)

If using QUBEKit multiple times per day with the same molecule, it is not necessary to update the 'run number'.
Not updating the run number when analysing the same molecule on the same day will lead to a backup being created.
If this is not wanted, you must update the run number with the command:

    -log Prop1201
    
where `Prop1201` is an example string which can be almost anything you like (no spaces or special characters).

**Inputs are not sanitised so code injection is possible but given QUBEKit's use occurs locally, you're only hurting yourself!
If you don't understand this, don't worry, just use alphanumeric log names like above.**

### QUBEKit Commands: High Throughput

Bulk commands are for high throughput analysis; they are invoked with the `-bulk` keyword.
A csv must be used when running a bulk analysis.
If you would like to generate a blank csv config file, simply run the command:

    QUBEKit -csv example.csv
    
where example.csv is the name of the config file you want to create.
This will automatically generate the file with the appropriate column headers.
The csv config file will be put into wherever you ran the command from.
When writing to the csv file, append rows after the header row, rather than overwriting it.

If you want to limit the number of molecules per csv file, simply add an argument to the command.
For example, if you have 23 pdb files and want to analyse them 12 at a time, use the command:

    QUBEKit -csv example.csv 12
    
This will generate two csv files, one with 12 molecules inside, the other with the remaining 11.
You can then fill in the rest of the csv as desired, or run immediately with the defaults.

Before running a bulk analysis, fill in each column for each molecule*; 
importantly, different config files can be supplied for each molecule.

*Only the name column needs to be filled (which is filled automatically with the generated csv),
any empty columns will simply use the default values:

* If the charge column is empty, charge will be set to 0;
* If the multiplicity column is empty, multiplicity will be set to 1; 
* If the config column is empty, the default config is used;
* The smiles string column only needs to be filled if a pdb is *not* supplied;
* Leaving the restart column empty will start the program from the beginning;
* Leaving the end column empty will end the program after a full analysis.

A bulk analysis is called with the `-bulk` command, followed by the name of the csv file:

    QUBEKit -bulk example.csv
    
Any pdb files should all be in the same place: where you're running QUBEKit from.
Upon executing this bulk command, QUBEKit will work through the rows in the csv file.
Each molecule will be given its own directory and log file (the same as single molecule analyses).

Please note, there are deliberately two config files.
The changeable parameters are spread across a .csv and a .ini config files.
The configs in the .ini are more likely to be kept constant across a bulk analysis.
For this reason, the .csv config contains highly specific parameters such as torsions which will change molecule to molecule.
The .ini contains more typically static parameters such as the basis sets and engines being used (e.g. PSI4, Chargemol, etc).
If you would like the ini config to change from molecule to molecule, you may specify that in the csv config.

You can change defaults inside the terminal when running bulk analyses, and these changed defaults will be printed to the log file.
However, the config files themselves will not be overwritten.
It is therefore recommended to manually edit the config files rather than doing, for example:

    QUBEKit -bulk example.csv -log run42 -ddec 3 -solvent true
    
Be aware that the names of the pdb files are used as keys to find the configs.
So, each pdb being analysed should have a corresponding row in the csv file with the correct name
(if using smiles strings, the name column will just be the name given to the created pdb file).

For example (csv row order does not matter, and you do not need to include smiles strings when a pdb is provided; column order *does* matter):

    <location>/:
        benzene.pdb
        ethane.pdb
        bulk_example.csv

    bulk_example.csv:
        name,charge,multiplicity,config,smiles,torsion_order,restart,end
        methane,,,default_config,C,,,
        benzene,,,default_config,,,,
        ethane,,,default_config,,,,

### QUBEKit Commands: Custom Start and End Points (single molecule)

QUBEKit also has the ability to run partial analyses, or redo certain parts of an analysis.
For a single molecule analysis, this is achieved with the `-end` and `-restart` commands. 

The stages are:

| Stage | Description |
| :---: | --- |
| *parametrise* | The molecule is parametrised using OpenFF, AnteChamber or an xml file. This step also loads in the molecule and extracts key information like the atoms and their coordinates. |
| *mm_optimise* | This is a quick, preliminary optimisation which speeds up later optimisations. (This stage is skippable.) |
| *qm_optimise* | This is the main quantum mechanical optimisation stage; default method is to use PSI4 with GeomeTRIC. |
| *hessian* | This again uses PSI4 or Gaussian to calculate the Hessian matrix which is needed for calculating bonding parameters. |
| *mod_sem* | Using the Hessian matrix, the bond lengths, angles and force constants are calculated with the Modified Seminario Method. |
| *density* | The density is calculated using Gaussian09. This is where the solvent is applied as well (if configured). | 
| *charges* | The charges are partitioned and calculated using Chargemol with DDEC3 or 6. Virtual sites are added where the error in ESP would be large without them. |
| *lennard_jones* | The charges are extracted and Lennard-Jones parameters (sigma and epsilon) are calculated. |
| *torsion_scan* | Using the molecule's geometry, a torsion scan is performed. The molecule can then be optimised with respect to these parameters. |
| *torsion_optimise* | The fitting and optimisation step for the torsional analysis. |
| *finalise* | This step (which is always performed, regardless of end-point) produces an xml file for the molecule. This stage also prints the final information to the log file and a truncated version to the terminal. |

In a normal run, all of these stages are executed sequentially,
but with `-end`, `-restart` and `-skip` you are free to run *from* any step *to* any step inclusively while skipping any non-essential steps.

When using `-end`, specify the end-point in the proceeding command (default `finalise`).

When using `-restart`, specify the start-point in the proceeding command (default `parametrise`).

When using `-skip`, specify any stage(s) which you needn't run. Be aware some stages depend on one another and cannot necessarily be skipped.

When using these commands, all other config-changing commands can be used in the same ways as before. For example:

**Beware skipping steps which are required for other stages of the analysis.**

```shell
# Inside the folder from a previous execution, 
# restart the qm optimisation and end having calculated the density.
QUBEKit -restart qm_optimise -end density
```
```shell
# Start a full parametrisation of benzene, 
# skip ahead to the qm optimisation and end before analysing the torsions.
QUBEKit -i benzene.pdb -log BEN001 -skip mm_optimise -end lennard_jones
```

If using `-end` but not `-restart`, a new directory and log file will be created within wherever the command is run from.
Just like a normal analysis.

However, using `-restart` requires files and other information from previous executions.
Therefore, `-restart` can only be run from *inside* a directory with those files present.

*Note: you do not need to use the `-i` (input file) command when restarting, QUBEKit will detect the pdb file for you.*

To illustrate this point, a possible use case would be to perform a full calculation on the molecule ethane,
then recalculate using a different (set of) default value(s):

```shell
QUBEKit -i ethane.pdb -log ETH001
...
cd QUBEKit_ethane_2019_01_01_ETH001
QUBEKit -restart density -end charges -ddec 3
```
Here, the calculation was performed with the default DDEC version 6, then rerun with version 3 instead, skipping over the early stages which would be unchanged.
It is recommended to copy (**not cut**) the directory containing the files because some of them will be overwritten when restarting.

Note that because `-restart` was used, it was not necessary to specify the pdb file name with `-i`.

### QUBEKit Commands: Custom Start and End Points (multiple molecules)

When using custom start and/or end points with bulk commands, the stages are written to the csv file, rather than the terminal.
If no start point is specified, a new working directory and log file will be created.
Otherwise, QUBEKit will find the correct directory and log file based on the log string and molecule name.
This means the **log string cannot be changed when restarting a bulk run**. 
There will however be a clear marker in the log file, indicating when an analysis was restarted.

Using a similar example as above, two molecules are analysed with DDEC6, then restarted for analysis with DDEC3:

    first_run.csv:
        name,charge,multiplicity,config,smiles,torsion_order,restart,end
        ethane,,,ddec6_config,,,,charges
        benzene,,,ddec6_config,,,,charges
    
    QUBEKit -bulk first_run.csv
    
    (optional: copy the folders produced to a different location to store results)
    
    
    second_run.csv:
        name,charge,multiplicity,config,smiles,torsion_order,restart,end
        ethane,,,ddec3_config,,,density,charges
        benzene,,,ddec3_config,,,density,charges
    
    QUBEKit -bulk second_run.csv

The first execution uses a config file for DDEC6 and runs from the beginning up to the charges stage.
The second execution uses a config file for DDEC3 and runs from the density stage to the charges stage.

### QUBEKit Commands: Checking Progress

Throughout an analysis, key information will be added to the log file.
This information can be quickly parsed by QUBEKit's `-progress` command

To display the progress of all analyses in your current directory and below, use the command:

    QUBEKit -progress
    
QUBEKit will find the log files in all QUBEKit directories and display a colour-coded table of the progress.  

| Indicator | Meaning |
| :---: | :---: |
| ✓ |  Completed successfully |
| ~ | Neither finished nor errored |
| S | Skipped |
| E | Started and failed for some reason |

Viewing the QUBEKit log file will give more information as to *why* it failed.

### QUBEKit Commands: Other Commands and Information

You cannot run multiple kinds of analysis at once. For example:

    QUBEKit -bulk example.csv -i methane.pdb -bonds g09
    
is not a valid command. These should be performed separately:

    QUBEKit -bulk example.csv
    QUBEKit -i methane.pdb -bonds g09
    
Be wary of running QUBEKit concurrently through different terminal windows.
The programs QUBEKit calls often just try to use however much memory is assigned in the config files;
this means they may try to take more than is available, leading to a hang-up, or--rarely--a crash.

## Cook Book

**Complete analysis of single molecule from its pdb file using only defaults:**

    QUBEKit -i molecule.pdb

All commands can be viewed by calling `QUBEKit -h`. Below is an explanation of what all these commands are:

| Command | Type | Description |
| --- | --- | --- |
| `-geo` | Bool: `True`, `true`, `t`, `False`, `false`, `f` | Enable or disable GeomeTRIC |
| `-ddec` | Int: `3`, `6` | Change DDEC version |
| `-solvent` | Bool | Enable or disable the solvent model |
| `-param` | String: `antechamber`, `openff`, `xml`  | Change the method for initial parametrisation |
| `-log` | String: `<arbitrary alphanumeric>` | Change the log file name and directory label |
| `-func` | String: `<any valid PSI4/Gaussian functional>` | Change the functional being used |
| `-basis` | String: `<any valid PSI4/Gaussian basis set>` | Change the basis set |
| `-vib` | Float: `0.0 - 1.0` | Change the vibrational scaling used with the basis set |
| `-memory` | Int: `1 - PC limit` | Change the amount of memory allocated |
| `-threads` | Int: `1 - PC limit` | Change the number of threads allocated |
| `-v` | Bool | Change the verbosity of the output to terminal |
| `-sites` | Bool | Enable or disable virtual sites |
| `-symmetry` | Bool | Enable or disable symmetrisation in the molecule parameters |
| `-display` | String: `<any valid QUBEKit object>` | Display a particular config option or parameter for this execution. e.g. `-display basis` would show the basis set used.

---

**Complete analysis of ethane from its smiles string using DDEC3, OpenFF and no solvent (`-log` command labels the analysis):**

    QUBEKit -sm CC -ddec 3 -param openff -solvent false -log ethane_example

---

**Analyse benzene from its pdb file until the charges are calculated; use DDEC3:**

    QUBEKit -i benzene.pdb -end charges -ddec 3 -log BENZ_DDEC3
    
**Redo that analysis but use DDEC6 instead:**

(Optional) Copy the folder and change the name to indicate it's for DDEC6:
    
    cp -r QUBEKit_benzene_2019_01_01_BENZ_DDEC3 QUBEKit_benzene_2019_01_01_BENZ_DDEC6

(Optional) Move into the new folder:

    cd QUBEKit_benzene_2019_01_01_BENZ_DDEC6

Rerun the analysis with the DDEC version changed.
This time we can restart just before the charges are calculated to save time.
Here we're restarting from density and finishing on charges:

    QUBEKit -restart density -end charges -ddec 6

This will still produce an xml in the `finalise` folder.

---

**Analyse methanol from its smiles string both with and without a solvent:**

    QUBEKit -sm CO -solvent true -log Methanol_Solvent

(Optional) Create and move into new folder

    cp -r QUBEKit_methanol_2019_01_01_Methanol_Solvent QUBEKit_methanol_2019_01_01_Methanol_No_Solvent
    cd QUBEKit_methanol_2019_01_01_Methanol_No_Solvent

    QUBEKit -solvent false -restart density

---

**Calculate the density for methane, ethane and propane using their pdbs:**

Generate a blank csv file with a relevant name:

    QUBEKit -csv density.csv
    
Fill in each row like so:

    density.csv:
        name,charge,multiplicity,config,smiles,torsion_order,restart,end
        methane,,,master_config.ini,,,,density
        ethane,,,master_config.ini,,,,density
        propane,,,master_config.ini,,,,density

Run the analysis:

    QUBEKit -bulk density.csv
    
Note, you can add more commands to the execution, but it is recommended that changes are made to the config files instead.

---

**Running the same analysis but using the smiles strings instead; this time do a complete analysis:**

Generate a blank csv with the name `simple_alkanes`:

    QUBEKit -csv simple_alkanes.csv

Fill in the csv file like so:

    simple_alkanes.csv:
        name,charge,multiplicity,config,smiles,torsion_order,restart,end
        methane,,,master_config.ini,C,,,
        ethane,,,master_config.ini,CC,,,
        propane,,,master_config.ini,CCC,,,

Run the analysis:

    QUBEKit -bulk simple_alkanes.csv

---

**Just calculating charges for acetone:**

Skip hessian and mod_sem, the two stages used to calculate the bonds and angles,
then end the analysis after the charge calculation.

    QUBEKit -i acetone.pdb -skip hessian mod_sem -end charges
