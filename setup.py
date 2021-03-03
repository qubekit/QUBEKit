#!/usr/bin/env python3

from setuptools import find_packages, setup
import versioneer


with open("README.md", "r") as file:
    long_description = file.read()

setup(
    name='qubekit',
    description='Quantum mechanical bespoke force field parameter generation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/qubekit/QUBEKit',
    packages=find_packages(),
    include_package_data=True,
    author='Joshua Thomas Horton, Chris Ringrose',
    entry_points={
        'console_scripts': [
            'QUBEKit = QUBEKit.run:main',
            'qubekit = QUBEKit.run:main',
            'QUBEKit-pro = QUBEKit.proteins.protein_run:main',
            'QUBEKit-gui = QUBEKit.GUI.gui:main'
        ]
    },
    version=versioneer.get_version(),
    license='MIT',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Programming Language :: Python :: 3.6',
    ],
    python_requires='~=3.6',
)
