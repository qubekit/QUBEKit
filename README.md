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
    
After relaunching the terminal you should now be able to use QUBEKit. Try the following command to bring up a short summary of the available commands and a description of how QUBEKit works:

    python run.py -h QUBEKit

## Requirements:
* [Anaconda3](https://www.anaconda.com/download/)
* [Gaussian09](http://gaussian.com/)
### Other Python modules used:

Many packages come installed with the above requirements, however there are some which require further installation. All packages must be installed through conda and use conda as their path.
* psi4
* rdkit

## In Development

QUBEKit should currently be considered a work in progress. While it is stable we are constantly working to improve the code and increase the amount of compatible software. A user tutorial can be found on our Github [wiki](https://github.com/cole-group/QuBeKit/wiki) page. 
