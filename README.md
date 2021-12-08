# QUBEKit - *Qu*antum Mechanical *Be*spoke force field tool*kit*

#### **Newcastle University UK - Cole Group**

| **Status** | [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/qubekit/QUBEKit.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/qubekit/QUBEKit/context:python) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  [![CI](https://github.com/qubekit/QUBEKit/actions/workflows/CI.yaml/badge.svg)](https://github.com/qubekit/QUBEKit/actions/workflows/CI.yaml)  [![codecov](https://codecov.io/gh/qubekit/QUBEKit/branch/main/graph/badge.svg?token=E554IAATJC)](https://codecov.io/gh/qubekit/QUBEKit)|
| :------ | :------ |
| **Foundation** | [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![python](https://img.shields.io/badge/python-3.6%2C%203.7%2C%203.8%2C%203.9-blue.svg)](https://www.python.org/) [![platforms](https://anaconda.org/conda-forge/qubekit/badges/platforms.svg)]() [![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/qubekit?color=blue&logo=anaconda&logoColor=white)](https://anaconda.org/conda-forge/qubekit)|
| **Installation** | [![Anaconda-Server Badge](https://anaconda.org/conda-forge/qubekit/badges/installer/conda.svg)](https://anaconda.org/conda-forge/qubekit) [![Anaconda-Server Badge](https://anaconda.org/conda-forge/qubekit/badges/downloads.svg)](https://anaconda.org/conda-forge/qubekit) [![Anaconda-Server Badge](https://anaconda.org/conda-forge/qubekit/badges/latest_release_date.svg)](https://anaconda.org/conda-forge/qubekit) |

## Table of Contents

* [What is QUBEKit?](https://github.com/qubekit/QUBEKit#what-is-qubekit)
    * [Development](https://github.com/qubekit/QUBEKit#in-development)
* [Installation](https://github.com/qubekit/QUBEKit#installation)
    * [Requirements](https://github.com/qubekit/QUBEKit#requirements)
    * [Installing as Dev](https://github.com/qubekit/QUBEKit#installing-as-dev)
* [Help](https://github.com/qubekit/QUBEKit#help)
    * [Config Files](https://github.com/qubekit/QUBEKit#before-you-start-config-files)
    * [QUBEKit Commands](https://github.com/qubekit/QUBEKit#qubekit-commands-running-jobs)
        * [Running Jobs](https://github.com/qubekit/QUBEKit#qubekit-commands-running-jobs)
        * [High Throughput](https://github.com/qubekit/QUBEKit#qubekit-commands-high-throughput)
        * [Custom Start and End Points](https://github.com/qubekit/QUBEKit#qubekit-commands-custom-start-and-end-points-single-molecule)
            * [Single Molecules](https://github.com/qubekit/QUBEKit#qubekit-commands-custom-start-and-end-points-single-molecule)
            * [Multiple Molecules](https://github.com/qubekit/QUBEKit#qubekit-commands-custom-start-and-end-points-multiple-molecules)
        * [Checking Progress](https://github.com/qubekit/QUBEKit#qubekit-commands-checking-progress)
        * [Other Commands and Information](https://github.com/qubekit/QUBEKit#qubekit-commands-other-commands-and-information)

## What is QUBEKit?

[QUBEKit](https://blogs.ncl.ac.uk/danielcole/qube-force-field/) is a Python 3.6+ based force field derivation toolkit for Linux operating systems.
Our aims are to allow users to quickly derive molecular mechanics parameters directly from quantum mechanical calculations.
QUBEKit pulls together multiple pre-existing engines, as well as bespoke methods to produce accurate results with minimal user input.
QUBEKit aims to avoid fitting to experimental data where possible while also being highly customisable.

Users who have used QUBEKit to derive force field parameters should cite the following references (also given in the `CITATION.cff` file):

* [QUBEKit: Automating the Derivation of Force Field Parameters from Quantum Mechanics](https://pubs.acs.org/doi/full/10.1021/acs.jcim.8b00767)
* [Biomolecular Force Field Parameterization via Atoms-in-Molecule Electron Density Partitioning](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.6b00027)
* [Harmonic Force Constants for Molecular Mechanics Force Fields via Hessian Matrix Projection](https://pubs.acs.org/doi/10.1021/acs.jctc.7b00785)

### In Development

While QUBEKit is stable we are constantly working to improve the code and broaden its compatibilities. 

We use software written by many people;
if reporting a bug, please (to the best of your ability) make sure it is a bug with QUBEKit and not with a dependency.
We welcome any suggestions, issues or pull requests for additions or changes.

## Installation

QUBEKit is available through conda-forge; this is the recommended installation method.
Github has our latest version which will likely have newer features but may not be stable.

    # Recommended
    conda install -c conda-forge qubekit

---
    
    # Recommended for Developers (see below)
    git clone https://github.com/qubekit/qubekit.git
    cd <install location>
    python setup.py develop    

### Requirements

* [Anaconda3](https://www.anaconda.com/download/)

Download Anaconda from the above link and install with the linux command:

    ./Anaconda3<version>.sh

*You may need to use `chmod +x Anaconda3<version>.sh` to make it executable.*

We recommend you add conda to your .bashrc when prompted.

Optionally, you may choose to use Gaussian and Chargemol. 
Gaussian is an option for QM optimisations, hessian calculations and density calculations.
Chargemol is an option for charge partitioning. QUBEKit contains alternative approaches for these calculations.

* [Gaussian09](http://gaussian.com/)

Installation of Gaussian is likely handled by your institution.

* [Chargemol](https://sourceforge.net/projects/ddec/files/)

Chargemol can be downloaded and installed from a zip file in the above link. 
Following unzipping, please export the path of the executable like so:

    export CHARGEMOL_DIR="/<location>/chargemol_09_26_2017/"

Conda packages are included in the conda-forge install.
If there are dependency issues or version conflicts in your environment, packages can be installed individually.

    conda install -c conda-forge qubekit

The following table details some core requirements included in the conda install of QUBEKit.
If any packages are missing from the install or causing issues, this table shows how to get them.

| **Package** | Conda Install |
| :------ | :------ |
| [OpenForceField](https://openforcefield.org/) | `conda install -c conda-forge openff-toolkit` |
| [OpenMM](http://openmm.org/) | `conda install -c omnia openmm` |
| [PSI4](http://www.psicode.org/) | `conda install -c psi4 psi4` |
| [QCEngine](https://pypi.org/project/qcengine/) | `conda install -c conda-forge qcengine` |
| [RDKit](http://rdkit.org/) | `conda install -c rdkit rdkit` |
| [TorsionDrive](https://github.com/lpwgroup/torsiondrive) | `conda install -c conda-forge torsiondrive` |

Adding lots of packages can be a headache. If possible, install using Anaconda through the terminal.
This is generally safest, as Anaconda should deal with versions and conflicts in your environment.
Generally, conda packages will have the conda install command on their website or github.
As a last resort, either git clone them and install:

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
    git clone https://github.com/qubekit/qubekit.git
    
    # Re-install QUBEKit outside of conda
    cd qubekit
    python setup.py develop

## Help

Below is general help for most of the commands available in QUBEKit.
There is some short help available through the terminal (invoked with `--help`).
`--help` may be called for any command you are unsure of:

    qubekit --help

---

    qubekit bulk run --help

---

    qubekit config create --help

All necessary long-form help is within this document.

### Config files

QUBEKit has a lot of familiar settings which are used in production and changing these can result in very different force field parameters.
The settings are controlled using json style config files which are easy to edit.

QUBEKit ***does*** have a full suite of defaults built in. 
To create a config file template with the name "example", containing default options, run the command:

    qubekit config create example.json

Simply edit the sections as desired to change the method, basis set, memory etc.

This config file can then be used to execute a QUBEKit job using the `--config` or `-c` flag in the run command.

### QUBEKit Commands: Running Jobs

A single molecule job can be executed with the `run` command. 

Some form of molecule input must also be supplied; 
either a file (designated with the `--input-file` or `-i` flag), or a smiles string (`--smiles` or `-sm`). 

In the case of a smiles string, the molecule must also be named with the `--name` or `-n` flag.

    qubekit run -i methane.pdb

    qubekit run -i ethane.mol2

    qubekit run -sm CO -n methanol


### QUBEKit Commands: High Throughput

Bulk commands are for high throughput analyses; they are invoked with the `bulk` keyword.
A csv must be used when running a bulk analysis.
If you would like to generate a blank csv file, simply run the command:

    qubekit bulk create example.csv 
    
where example.csv is the name of the file you want to create.
This will automatically generate a csv file with the appropriate column headers.
When writing to the csv file, simply append rows after the header row with the relevant information required.*

Importantly, different config files can be supplied for each molecule in each row.

name,smiles,multiplicity,config_file,restart,end

*Only the name column needs to be filled (which is filled automatically with the generated csv),
any empty columns will simply use the default values:

* The smiles string column only needs to be filled if a pdb is *not* supplied;
* If the multiplicity column is empty, multiplicity will be set to 1; 
* If the config column is empty, the default config is used;
* Leaving the restart column empty will start the program from the beginning;
* Leaving the end column empty will end the program after a full analysis.

A bulk analysis is called with the `run` command, followed by the name of the csv file:

    qubekit bulk run example.csv
    
Any pdb files should all be in the same place: where you're running QUBEKit from.
Upon executing this bulk command, QUBEKit will work through the rows in the csv file.
Each molecule will be given its own directory (the same as single molecule analyses).

Be aware that the names of the pdb files are used as keys to find the configs.
So, each pdb being analysed should have a corresponding row in the csv file with the correct name
(if using smiles strings, the name column will just be the name given to the created pdb file).

For example (barring the header, csv row order does not matter, and you do not need to include smiles strings when a file is provided; column order *does* matter):

    <location>/:
        benzene.pdb
        ethane.mol2
        bulk_example.csv

    bulk_example.csv:
        name,smiles,multiplicity,config_file,restart,end
        methane,C,,default_config.json,,
        benzene,,,default_config.json,,
        ethane,,,default_config.json,,

### QUBEKit Commands: Custom Start and End Points (single molecule)

QUBEKit also has the ability to run partial analyses, or redo certain parts of an analysis.
For a single molecule analysis, this is achieved with the `--end` (`-e`) and `restart` commands. 

The stages are:

| Stage | Description |
| :---: | --- |
| *parametrisation* | The molecule is parametrised using OpenFF, AnteChamber or an xml file. This step also loads in the molecule and extracts key information like the atoms and their coordinates. |
| *optimisation* | There is a quick, preliminary optimisation which speeds up the following QM optimisation. |
| *hessian* | This uses the same QM engine as the prior optimisation to calculate the Hessian matrix which is needed for calculating bonding parameters. |
| *charges* | The electron density is calculated. This is where the solvent is applied as well (if configured). Next, the charges are partitioned and calculated |
| *virtual_sites* | Virtual sites are added where the error in ESP would be large without them. |
| *non_bonded* | Lennard-Jones parameters (sigma and epsilon values) are calculated. |
| *bonded_parameters* | Using the Hessian matrix, the bond lengths, angles and force constants are calculated with the Modified Seminario Method. | 
| *torsion_scanner* | Using the molecule's geometry, a torsion scan is performed. The molecule can then be optimised with respect to these parameters. |
| *torsion_optimisation* | The fitting and optimisation step for the torsional analysis. |

In a normal run, all of these stages are executed sequentially,
but with `--end`, `restart` and `--skip-stages` you are free to run *from* any step *to* any step inclusively while skipping any non-essential steps.

When using `--end` (or `-e`), specify the end-point in the proceeding command.

When using `restart`, specify the start-point in the proceeding command.

When using `--skip-stages` (or `-s`), specify any stage(s) which you needn't run. Be aware some stages depend on one another and cannot necessarily be skipped.


### QUBEKit Commands: Custom Start and End Points (multiple molecules)

When using custom start and/or end points with bulk commands, the stages are written to the csv file, rather than the terminal.
If no start point is specified, a new working directory be created and QUBEKit will run as normal.
Otherwise, QUBEKit will find the correct directory based on the molecule name.


### QUBEKit Commands: Checking Progress

Throughout an analysis, key information will be added to the run directory.
This information can be quickly parsed by QUBEKit's `progress` command

To display the progress of all analyses in your current directory and below, use the command:

    qubekit progress
    
QUBEKit will use the `workflow_result.json` which is in all QUBEKit directories and display a colour-coded table of the progress.  

| Indicator | Meaning |
| :---: | :---: |
| ✓ |  Completed successfully |
| ~ | Neither finished nor errored |
| S | Skipped |
| E | Started and failed for some reason |

Viewing the QUBEKit.err file will give more information as to *why* it failed.

### QUBEKit Commands: Other Commands and Information

You cannot run multiple kinds of analysis at once. For example:

    qubekit bulk run example.csv run -i methane.pdb
    
is not a valid command. These should be performed separately:

    qubekit bulk run example.csv
    qubekit run -i methane.pdb
    
Be wary of running QUBEKit concurrently through different terminal windows.
The programs QUBEKit calls often just try to use however much memory is assigned in the config files;
this means they may try to take more than is available, leading to a hang-up, or--rarely--a crash.
