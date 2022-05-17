Installation
============

Linux / MacOS
-------------

QUBEKit is available through conda-forge; this is the recommended installation method.
Github has our latest version which will likely have newer features but may not be stable.

.. code-block:: Bash

    # Recommended
    conda install -c conda-forge qubekit

Alternatively, you can install the dependencies manually in the env.yml and use pip to install the package:

.. code-block:: Bash

    git clone https://github.com/qubekit/qubekit.git
    cd qubekit
    python install .

..

    WARNING: Note the issue with PSI4 as the clash between the conda channels defaults and conda-forge.


Requirements
------------

We recommend using conda (https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)
to manage the dependencies.

TODO - PSI?
Optionally, you may choose to use Gaussian (http://gaussian.com/) and Chargemol.
Gaussian is an option for QM optimisations, hessian calculations and density calculations.
Chargemol is an option for charge partitioning.
QUBEKit contains alternative approaches for these calculations.

Minimal conda packages are included in the conda-forge install with all optional
engine packages left to the user to install. If there are dependency issues or
version conflicts in your environment, packages can be installed individually.

The following table details some optional calculation engine packages
which are not included in the QUBEKit conda package and how to install them.

===============  ================================================
     Package          Output
===============  ================================================
PSI4 >=1.4.1     conda install -c psi4 psi4
---------------  ------------------------------------------------
Chargemol        conda install -c conda-forge chargemol
---------------  ------------------------------------------------
xtb-python       conda install -c conda-forge xtb-python
---------------  ------------------------------------------------
torchani         conda install -c conda-forge torchani
===============  ================================================

Adding lots of packages can be a headache. If possible, install using Anaconda through the terminal.
This is generally safest, as Anaconda should deal with versions and conflicts in your environment.
Generally, conda packages will have the conda install command on their website or github.
As a last resort, either git clone them and install:

.. code-block:: Bash

    pip install git+https://github.com/qubekit/qubekit.git

or follow the described steps in the respective documentation.


Developer Installation
----------------------
If downloading QUBEKit to edit the latest version of the source code,
the easiest method is to install via conda, then remove the conda version
of qubekit and git clone. This is accomplished with a few simple commands:

.. code-block:: Bash

    # Install QUBEKit as normal
    conda install -c conda-forge qubekit

    # Remove ONLY the QUBEKit package itself, leaving all dependencies installed
    # and on the correct version
    conda remove --force qubekit

    # Re-download the latest QUBEKit through github
    git clone https://github.com/qubekit/qubekit.git

    # Re-install QUBEKit
    cd qubekit
    pip install .
