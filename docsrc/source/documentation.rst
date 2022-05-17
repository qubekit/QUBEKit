Documentation
=============

QUBEKit's primary interface is the command line. 

Here we present some examples.

Running Jobs
------------

A single molecule job can be executed with the `run` command.

Some form of molecule input must also be supplied; either a file
(designated with the `--input-file` or `-i` flag), or a smiles string (`--smiles` or `-sm`).

In the case of a smiles string, the molecule must also be named with the `--name` or `-n` flag.

.. code-block:: Bash

    qubekit run -i methane.pdb

    qubekit run -i ethane.mol2

    qubekit run -sm CO -n methanol


Config files
------------
QUBEKit has a lot of familiar settings which are used in production and changing
these can result in very different force field parameters. The settings are
controlled using json style config files which are easy to edit.

QUBEKit does have a full suite of defaults built in. To create a config file
template with the name "example", containing default options, run the command:

.. code-block:: Bash

    qubekit config create example.json

Simply edit the sections as desired to change the method, basis set, memory etc.

This config file can then be used to execute a QUBEKit job using the `--config` or `-c`
flag in the run command.


Custom Start and End Points
---------------------------

Single molecule
~~~~~~~~~~~~~~~

QUBEKit also has the ability to run partial analyses, or redo certain parts of an analysis.
For a single molecule analysis, this is achieved with the `--end` (`-e`) and `restart` commands.

The stages are:

========================  ======================
Stage                     Description
========================  ======================
parametrisation           | The molecule is parametrised using OpenFF, AnteChamber or an xml file.
                          | This step also loads in the molecule and extracts key information like
                          | the atoms and their coordinates.
optimisation              | There is a quick, preliminary optimisation which speeds up the
                          | following QM optimisation.
hessian                   | This uses the same QM engine as the prior optimisation to calculate
                          | the Hessian matrix which is needed for calculating bonding parameters.
charges                   | The electron density is calculated. This is where the solvent is
                          | applied as well (if configured). Next, the charges are partitioned
                          | and calculated.
virtual_sites             | Virtual sites are added where the error in ESP would be large
                          | without them.
non_bonded                | Lennard-Jones parameters (sigma and epsilon values) are calculated.
bonded_parameters         | Using the Hessian matrix, the bond lengths, angles and force constants
                          | are calculated with the Modified Seminario Method.
torsion_scanner           | Using the molecule's geometry, a torsion scan is performed. The molecule
                          | can then be optimised with respect to these parameters.
torsion_optimisation      | The fitting and optimisation step for the torsional analysis.
========================  ======================

In a normal run, all of these stages are executed sequentially,
but with `--end`, `restart` and `--skip-stages` you are free to run *from* any step *to* any step
inclusively while skipping any non-essential steps.

When using `--end` (or `-e`), specify the end-point in the proceeding command.

When using `restart`, specify the start-point in the proceeding command.

When using `--skip-stages` (or `-s`), specify any stage(s) which you needn't run. Be aware some stages
depend on one another and cannot necessarily be skipped.


Multiple molecules
~~~~~~~~~~~~~~~~~~

When using custom start and/or end points with bulk commands, the stages are written to the csv file, rather than the terminal.
If no start point is specified, a new working directory be created and QUBEKit will run as normal.
Otherwise, QUBEKit will find the correct directory based on the molecule name.


Checking Progress
-----------------

Throughout an analysis, key information will be added to the run directory.
This information can be quickly parsed by QUBEKit's `progress` command

To display the progress of all analyses in your current directory and below, use the command:

    qubekit progress

QUBEKit will use the `workflow_result.json` which is in all QUBEKit directories and display a colour-coded table of the progress.

======================================  ======================================
Indicator                               Meaning
======================================  ======================================
✓                                       Completed successfully 
~                                       Neither finished nor errored
S                                       Skipped
E                                       Started and failed for some reason
======================================  ======================================

Viewing the QUBEKit.err file will give more information as to **why** it failed.


High Throughput
---------------

Bulk commands are for high throughput analyses; they are invoked with the bulk keyword.
A csv must be used when running a bulk analysis. If you would like to generate a
blank csv file, simply run the command:

.. code-block:: Bash

    qubekit bulk create example.csv


where example.csv is the name of the file you want to create.
This will automatically generate a csv file with the appropriate column headers.
When writing to the csv file, simply append rows after the header row with the
relevant information required.*

Importantly, different config files can be supplied for each molecule in each row.

.. code-block:: Bash

    name,smiles,multiplicity,config_file,restart,end

*Only the name column needs to be filled (which is filled automatically with the generated csv),
any empty columns will simply use the default values:

- The smiles string column only needs to be filled if a file is not supplied;
- If the multiplicity column is empty, multiplicity will be set to 1;
- If the config column is empty, the default config is used;
- Leaving the restart column empty will start the program from the beginning;
- Leaving the end column empty will end the program after a full analysis.


A bulk analysis is called with the run command, followed by the name of the csv file:

.. code-block:: Bash

    qubekit bulk run example.csv

Any necessary files should all be in the same place: where you're running QUBEKit from.
Upon executing this bulk command, QUBEKit will work through the rows in the csv file.
Each molecule will be given its own directory (the same as single molecule analyses).

Be aware that the names of the files are used as keys to find the configs.
So, each molecule file being analysed should have a corresponding row in
the csv file with the correct name (if using smiles strings, the name
column will just be the name given to the created file).

For example (barring the header, csv row order does not matter, and you do not
need to include smiles strings when a file is provided; column order does matter):

    <location/>:
        benzene.pdb
        ethane.mol2
        bulk_example.csv

    bulk_example.csv:
        name,smiles,multiplicity,config_file,restart,end
        methane,C,,default_config.json,,
        benzene,,,default_config.json,,
        ethane,,,default_config.json,,


Other
-----
You cannot run multiple kinds of analysis at once. For example:

.. code-block:: Bash

    qubekit bulk run example.csv run -i methane.pdb

is not a valid command. These should be performed separately:

.. code-block:: Bash

    qubekit bulk run example.csv
    qubekit run -i methane.pdb

Be wary of running QUBEKit concurrently through different terminal windows.
The programs QUBEKit calls often just try to use however much memory is assigned in the config files;
this means they may try to take more than is available, leading to a hang-up, or a crash.


Help
----

Below is general help for most of the commands available in QUBEKit.
There is some short help available through the terminal (invoked with `--help`).
`--help` may be called for any command you are unsure of:

.. code-block:: Bash

    qubekit --help
    qubekit bulk run --help
    qubekit config create --help

All necessary long-form help is within this document.