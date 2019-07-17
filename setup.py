# !/usr/bin/env python3

import os
from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), encoding='utf-8') as file:
    long_description = file.read()

setup(
    name='qubekit',
    description='Quantum mechanical bespoke force field parameter generation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/qubekit/QUBEKit',
    packages=find_packages(),
    include_package_data=True,
    author=['Joshua Thomas Horton', 'Chris Ringrose'],
    entry_points={
        'console_scripts': [
            'QUBEKit = QUBEKit.run:main',
            'qubekit = QUBEKit.run:main',
            'QUBEKit-pro = QUBEKit.protein_run:main',
            'QUBEKit-gui = QUBEKit.GUI.gui:main'
        ]
    },
    version='2.4.0',
    license='MIT',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Programming Language :: Python :: 3.6',
    ],
    python_requires='~=3.6'
)

