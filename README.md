# QUBEKit - QUantum BEspoke force field toolKit

**Newcastle University Cole Group.**

## Table of Contents

* [What is QUBEKit?](https://github.com/jthorton/QUBEKitdev#what-is-qubekit)
    * [Development](https://github.com/jthorton/QUBEKitdev#in-development)
* [Installation](https://github.com/jthorton/QUBEKitdev#installation)
    * [Requirements](https://github.com/jthorton/QUBEKitdev#requirements)
* [Help](https://github.com/jthorton/QUBEKitdev#help)
    * [Before You Start: Config Files](https://github.com/jthorton/QUBEKitdev#before-you-start-config-files)
    * [QUBEKit Commands](https://github.com/jthorton/QUBEKitdev#qubekit-commands-running-jobs)
        * [Running Jobs](https://github.com/jthorton/QUBEKitdev#qubekit-commands-running-jobs)
        * [Some Examples](https://github.com/jthorton/QUBEKitdev#qubekit-commands-some-examples)
        * [Logging](https://github.com/jthorton/QUBEKitdev#qubekit-commands-some-examples)
        * [High Throughput](https://github.com/jthorton/QUBEKitdev#qubekit-commands-high-throughput)
        * [Custom Start and End Points](https://github.com/jthorton/QUBEKitdev#qubekit-commands-custom-start-and-end-points-single-molecule)
            * [Single Molecules](https://github.com/jthorton/QUBEKitdev#qubekit-commands-custom-start-and-end-points-single-molecule)
            * [Multiple Molecules](https://github.com/jthorton/QUBEKitdev#qubekit-commands-custom-start-and-end-points-multiple-molecules)
        * [Other Commands and Information](https://github.com/jthorton/QUBEKitdev#qubekit-commands-other-commands-and-information)
* [Cook Book](https://github.com/jthorton/QUBEKitdev#cook-book)


## What is QUBEKit?

[QUBEKit](https://blogs.ncl.ac.uk/danielcole/qube-force-field/) is a Python based force field derivation toolkit.
It aims to allow users to quickly derive molecular mechanics parameters directly from quantum mechanical calculations.
QUBEKit pulls together multiple pre-existing engines, as well as bespoke methods to produce accurate results with minimal user input.
QUBEKit aims to use as few parameters as possible while also being highly customisable.

Users who have used QUBEKit to derive any new force field parameters should cite the following papers:

* [Biomolecular Force Field Parameterization via Atoms-in-Molecule Electron Density Partitioning](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.6b00027)
* [Harmonic Force Constants for Molecular Mechanics Force Fields via Hessian Matrix Projection](https://pubs.acs.org/doi/10.1021/acs.jctc.7b00785)

### In Development

QUBEKit should currently be considered a work in progress.
While it is stable we are constantly working to improve the code and broaden its compatibility. 
We use lots of software written by many different people;
if reporting a bug please (to the best of your ability) make sure it is a bug with QUBEKit and not with a dependency.

## Installation

To install, it is recommended to clone the QUBEKit folder into a home directory and run the setup.py script: 

    git clone git@github.com:cole-group/QuBeKit.git
    cd QUBEKit
    python setup.py install

### Requirements

* [Anaconda3](https://www.anaconda.com/download/)
* [Gaussian09](http://gaussian.com/)
* [PSI4](http://www.psicode.org/)

### Other Python modules used

Many packages come installed with the above requirements, however there are some which require further installation. All packages must be installed through conda and use conda as their path.
(Standard library packages and conda default packages (scipy, numpy, etc) are not listed.)

* [GeomeTRIC](https://github.com/leeping/geomeTRIC)
* [RDKit](http://rdkit.org/)
* [OpenFF](https://openforcefield.org/)

You should now be able to use QUBEKit straight away from the command line or as an import.

Adding lots of packages can be a headache. If possible, install using Anaconda through the terminal.
This is generally safest, as Anaconda will deal with versions and conflicts in your environment.
Generally, conda packages will have the conda install command on their website/github.
For the software not available through Anaconda, either git clone them and install:

    git clone http://<git_address_here>
    python setup.py install

or follow the described steps in the respective documentation.

## Help

Below is general help for most of the commands available in QUBEKit.
Please be aware there are currently no terminal help commands; all necessary information is within this document.

### Before you start: Config files

QUBEKit has a lot of settings which are used in production and changing these can result in very different force field parameters.
The settings are controlled using ini style config files which are easy to edit.
After installation you should notice a ```QUBEKit_configs``` 
folder in your main home directory; now you need to create a master template.
To do this use the command ```QUBEKit -setup``` where you will then be presented with some options:

    You can now edit config files using QUBEKit, chose an option to continue:
    1)Edit a config file
    2)Create a new master template
    3)Make a normal config file

Choose option two to set up a new template which will be used every time you run QUBEKit 
(unless you supply the name of another ini file in the configs folder).
The only parameter that must be changed for QUBEKit to run is the chargemol path in the descriptions section.
This option is what controls where the chargemol code is kept on your PC.
It should be the location of the chargemol home directory, plus the name of the chargemol folder itself:

    '/home/<user>/Programs/chargemol_09_26_2017'

### QUBEKit Commands: Running Jobs

Given a list of commands, such as: "-end", "psi4" some are taken as single word commands.
Others however, such as changing defaults: ("-c", "0") ("-m", "1") etc are taken as tuple commands.
The first command of tuple commands is always preceded by a "-", while single word commands are not.
(An error is raised for hanging commands e.g. "-c", "1", "-sm".)

Single-word commands are taken in any order, assuming they are given, tuple commands are taken as ("-key", "val").
All commands are optional. If nothing at all is given, the program will run entirely with defaults.

Files to be analysed must be written with their file extension (.pdb) attached or they will not be recognised commands.
All commands should be given in lower case with two main exceptions;
you may use whatever case you like for the name of files (e.g. DMSO.pdb) or log strings (e.g. Run013).

### QUBEKit Commands: Some Examples

Running a full analysis on molecule.pdb with a non-default charge of 1, the default charge engine (chargemol) and with geometric off:
Note, ordering does not matter as long as tuples commands (-c, 1) are together.
    
    QUBEKit molecule.pdb -c 1 -geo false
    QUBEKit -c 1 -geo false molecule.pdb

Running a full analysis with a non-default engine (g09):

    QUBEKit molecule.pdb -bonds g09

The program will tell the user which defaults are being used, and which commands were given.
Errors will be raised for any invalid commands and the program will not run.

Try running QUBEKit with the command:

    QUBEKit -sm C -end hessian

This will generate a methane pdb file (and mol file) using its smiles string: "C",
then QUBEKit will analyse it until the hessian is calculated.
See "QUBEKit Commands: Custom Start and End Points (single molecule)" below for more details on "-end".

## QUBEKit Commands: Logging

Each time QUBEKit runs, a new working directory containing a log file will be created.
The name of the directory will contain the run number provided via the terminal command "-log" (or the default run number in the configs if none is provided).
This log file will store which methods were called, how long they took, and any docstring for them (if it exists).
The log file will also contain information regarding the config options used, as well as the commands given and much more.
The log file updates in real time and contains far more information than is printed to the terminal during a run.
If there is an error with QUBEKit, the full stack trace of an exception will be stored in the log file.

Many errors have custom exceptions to help if, for example, a module has not been installed correctly.

The format for the name of the active directory is:

    QUBEKit_moleculename_YYYY_MM_DD_runnumber
    
The format for the log file (which is inside this directory) is similar:

    QUBEKit_log_moleculename_YYYY_MM_DD_runnumber

If using QUBEKit multiple times per day with the same molecule, it is therefore necessary to update the 'run number'.
Not updating the run number when analysing the same molecule on the same day will prevent the program from running.
This is to prevent the directory being overwritten.

Updating the run number can be done with the command:

    -log Prop1201
    
where 'Prop1201' is an example string which can be almost anything you like (no spaces or special characters).

**Inputs are not sanitised so code injection is possible but given QUBEKit's use occurs locally, you're only hurting yourself!
If you don't understand this, don't worry, just use alphanumeric log names like above.**

### QUBEKit Commands: High Throughput

Bulk commands are for high throughput analysis; they are invoked with the -bulk keyword.
A csv must be used when running a bulk analysis.
If you would like to generate a blank csv config file, simply run the command:

    QUBEKit -csv example.csv
    
where example.csv is the name of the config file you want to create.
This will automatically generate the file with the appropriate column headers.
The csv config file will be put into wherever you ran the command from.
When writing to the csv file, append rows after the header row, rather than overwriting it.

Before running a bulk analysis, fill in each column for each molecule*; 
importantly, different config files can be supplied for each molecule.

*Not all columns need to be filled:

* If the config column is not filled, the default config is used.
* The smiles string column only needs to be filled if a pdb is not supplied.
* Leaving the start column empty will start the program from the beginning.
* Leaving the end column empty will end the program after a full analysis.

A bulk analysis is called with the command:

    QUBEKit -bulk example.csv
    
where example.csv is the csv file containing all the information described above.

Any pdb files should all be in the same place: where you're running QUBEKit from.
Upon executing this bulk command, QUBEKit will work through the rows in the csv file.
Each molecule will be given its own directory and log file (as in non-bulk executions).

Please note, there are deliberately two config files.
The changeable parameters are spread across a .csv and a .ini config files.
The configs in the .ini are more likely to be kept constant across a bulk analysis.
For this reason, the .csv config contains highly specific parameters such as torsions which will change molecule to molecule.
The .ini contains more typically static parameters such as the basis sets and engines being used (e.g. psi4, chargemol, etc).
If you would like the .ini config to change from molecule to molecule, you may specify that in the .csv config.

You can change defaults inside the terminal when running bulk analyses, and these changed defaults will be printed to the log file.
However, the config files themselves will not be overwritten.
It is therefore recommended to manually edit the config files rather than doing, for example:

    QUBEKit -bulk pdb example.csv -log run42 -ddec 3 -solvent true
    
Be aware that the names of the pdb files are used as keys to find the configs.
So, each pdb being analysed should have a corresponding row in the csv file with the correct name.

For example (csv row order does not matter, and you do not need to include smiles strings when a pdb is provided):

    <location>/:
        benzene.pdb
        ethane.pdb
        bulk_example.csv

    bulk_example.csv:
        name,charge,multiplicity,config,smiles string,torsion order,start,end
        methane,0,1,default_config,C,,,
        benzene,0,1,default_config,,,,
        ethane,0,1,default_config,,,,

### QUBEKit Commands: Custom Start and End Points (single molecule)

QUBEKit also has the ability to run partial analyses, or redo certain parts of an analysis.
For single molecule analysis, this is achieved with the -end and -restart commands. 

-end specifies the final stage for an analysis, where 'finalise' is the default. The stages are:

* rdkit_optimise - This is a quick, preliminary optimisation with RDKit which speeds up later optimisations.
This step also loads in the molecule and extracts key information like the atoms and their coordinates. 
* parametrise - The molecule is parametrised using OpenFF, AnteChamber or XML.
* qm_optimise - This is the main optimisation stage, default method is to use PSI4 with GeomeTRIC.
* hessian - This again uses PSI4 to calculate the Hessian matrix.
* mod_sem - Using the Hessian matrix, the bonds and angles terms are calculated with the Modified Seminario Method.
* density - The density is calculated using Gaussian09. This is where the solvent is applied as well. 
* charges - The charges are partitioned and calculated using Chargemol and DDEC3 or 6.
* lennard_jones - The charges are extracted ready for producing an XML.
The Lennard-Jones parameters are also calculated and stored.
* torsions - Using the molecule's geometry, a torsion scan is performed.
The molecule can then be optimised with respect to these parameters.
* finalise - This step (which is always performed, regardless of end-point) produces an XML for the molecule.
This stage also prints the final information to the log file and a truncated version to the terminal.

In a normal run, all of these functions are called sequentially,
but with -end and -restart you are free to run *from* any step *to* any step inclusively.

When using -end, simply specify the end-point in the proceeding command,
when using -restart, specify the start-point, then the end-point as space separated commands.
When using these commands, all other commands can be used in the same ways as before. For example:

    QUBEKit benzene.pdb -log BEN001 -end charges -geo false 
    QUBEKit ethanol.pdb -restart hessian torsions -ddec 3
    
If specifying a different end-point, a new directory and log file will be created within wherever the command is run from.

Because changing the start point requires files and other information from previous executions, this can only be run from
inside a directory with those files present.

To illustrate this point, a possible use case would be to perform a full calculation on the molecule ethane,
then recalculate using a different (set of) default value(s):

    QUBEKit ethane.pdb -log ETH001
    ...
    cd QUBEKit_ethane_2019_01_01_ETH001
    QUBEKit ethane.pdb -restart density charges -ddec 3
    
Here, the calculation was performed with the default DDEC version 6, then rerun with version 3 instead, skipping over the early stages which would be unchanged.
It is recommended to copy (**not cut**) the directory containing the files because some of them will be overwritten when restarting.

### QUBEKit Commands: Custom Start and End Points (multiple molecules)

When using custom start and/or end points with bulk commands, the stages are written to the csv file, rather than the terminal.
If no start point is specified, a new working directory and log file will be created.
Otherwise, QUBEKit will find the correct directory and log file based on the log string and molecule name.
This means the **log string cannot be changed when restarting a bulk run**.

Using a similar example as above, two molecules are analysed with DDEC6, then restarted for analysis with DDEC3:

    first_run.csv:
        name,charge,multiplicity,config,smiles string,torsion order,start,end
        ethane,0,1,ddec6_config,,,,charges
        benzene,0,1,ddec6_config,,,,charges
    
    QUBEKit -bulk first_run.csv
    
    (optional: copy the folders produced to a different location to store results)
    
    
    second_run.csv:
        name,charge,multiplicity,config,smiles string,torsion order,start,end
        ethane,0,1,ddec3_config,,,density,charges
        benzene,0,1,ddec3_config,,,density,charges
    
    QUBEKit -bulk second_run.csv

The first execution uses a config file for DDEC6 and runs from the beginning up to the charges stage.
The second execution uses a config file for DDEC3 and runs from the density stage to the charges stage.

### QUBEKit Commands: Other Commands and Information

To display the progress of all analyses in your current directory:

    QUBEKit -progress
    
This will scan where you are for all directories starting with 'QUBEKit'.
QUBEKit will then find the log files in those directories and display a table of the progress indicated in those log files.  

You cannot run multiple kinds of analysis at once. For example:

    QUBEKit -bulk example.csv methane.pdb -bonds g09
    
is not a valid command. These should be performed separately:

    QUBEKit -bulk example.csv
    QUBEKit methane.pdb -bonds g09
    
Be wary of running QUBEKit concurrently through different terminal windows.
The programs QUBEKit calls often just try to use however much RAM is assigned in the config files;
this means they may try to take more than is available, leading to a crash.


## Cook Book

**Complete analysis of single molecule from its pdb file using only defaults:**

    QUBEKit molecule.pdb

Optional commands:

* Enable or disable geomeTRIC: 
```-geo true``` or ```-geo false```

* Change DDEC version: 
```-ddec 3``` or ```-ddec 6```

* Enable or disable the solvent model: 
```-solvent true``` or ```-solvent false```

* Change the method for initial parametrisation: 
```-param openff```, ```-param xml```, ```-param antechamber```

* Change the log file name and directory label:
```-log Example123```

* Change the functional being used:
```-func B3LYP```

* Change the basis set:
```-basis 6-31G```

**Complete analysis of ethane from its smiles string using DDEC3, OpenFF and no solvent:**

    QUBEKit -sm CC -ddec 3 -param openff -solvent false -log ethane_example

**Analyse benzene from its pdb file until the charges are calculated; use DDEC3 (```-log``` command labels the analysis):**

    QUBEKit benzene.pdb -end charges -ddec 3 -log BENZ_DDEC3
    
**Redo that analysis but use DDEC6 instead:**

*If you don't care about overwriting the previous analysis, skip the next two steps.*

Copy the folder and change the name to indicate it's for DDEC6:
    
    cp -r QUBEKit_benzene_2019_01_01_BENZ_DDEC3 QUBEKit_benzene_2019_01_01_BENZ_DDEC6

Move into the new folder:

    cd QUBEKit_benzene_2019_01_01_BENZ_DDEC6

Rerun the analysis with the DDEC version changed.
This time we can restart just before the charges are calculated to save time.
Here we're restarting from density and finishing on charges:

    QUBEKit -restart density charges -ddec 6

**Analyse methanol from its smiles string both with and without a solvent:**

    QUBEKit -sm CO -solvent true -log Methanol_Solvent

*Again, skip the following two steps if you don't care about files being overwritten.*

    cp -r QUBEKit_methanol_2019_01_01_Methanol_Solvent QUBEKit_methanol_2019_01_01_Methanol_No_Solvent
    cd QUBEKit_methanol_2019_01_01_Methanol_No_Solvent

    QUBEKit -solvent false -restart density finalise
    
**Calculate the density for methane, ethane and propane using their pdbs:**

Generate a blank csv file with a relevant name:

    QUBEKit -csv density.csv
    
Fill it in like so:

    density.csv:
        name,charge,multiplicity,config,smiles string,torsion order,start,end
        methane,0,1,master_config.ini,,,,density
        ethane,0,1,master_config.ini,,,,density
        propane,0,1,master_config.ini,,,,density

Run the analysis:

    QUBEKit -bulk density.csv
    
Note, you can add more commands to the execution but it is recommended that changes are made to the config files instead.

**Running the same analysis but using the smiles strings instead; this time do a complete analysis:**

Generate a blank csv:

    QUBEKit -csv simple_alkanes.csv

Fill in the csv file like so:

    simple_alkanes.csv:
        name,charge,multiplicity,config,smiles string,torsion order,start,end
        methane,0,1,master_config.ini,C,,,
        ethane,0,1,master_config.ini,CC,,,
        propane,0,1,master_config.ini,CCC,,,

Run the analysis:

    QUBEKit -bulk simple_alkanes.csv