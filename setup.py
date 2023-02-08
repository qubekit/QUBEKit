"""
QUBEKit
Quantum mechanical bespoke force field parameter generation.
"""

from setuptools import find_packages, setup

import versioneer

short_description = __doc__.split("\n")

try:
    with open("README.md", "r") as file:
        long_description = file.read()
except IOError:
    long_description = "\n".join(short_description[2:])

setup(
    name="qubekit",
    description=short_description[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qubekit/QUBEKit",
    packages=find_packages(),
    include_package_data=True,
    author="Joshua Thomas Horton, Chris Ringrose",
    entry_points={
        "console_scripts": [
            "QUBEKit = qubekit.cli.cli:cli",
            "qubekit = qubekit.cli.cli:cli",
            "QUBEKit-pro = qubekit.proteins.protein_run:main",
        ],
        "openff.toolkit.plugins.handlers": [
            "QUBEKitvdWTS = qubekit.cli.utils:QUBEKitHandler",
            "LocalCoordinateVirtualSites = qubekit.cli.utils:LocalCoordinateVirtualSiteHandler",
            "UreyBradley = qubekit.cli.utils:UreyBradleyHandler",
            "ProperRyckhaertBellemans = qubekit.cli.utils:ProperRyckhaertBellemansHandler",
            "BondChargeCorrection = qubekit.cli.utils:BondChargeCorrectionHandler"
        ]
    },
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license="MIT",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Programming Language :: Python :: 3.6",
    ],
    python_requires=">=3.6",
)
