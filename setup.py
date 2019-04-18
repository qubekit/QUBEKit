# !/usr/bin/env python

from setuptools import setup, find_packages
from pathlib import Path
from os import path, makedirs
from datetime import datetime

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as file:
    long_description = file.read()

setup(
    name='qubekit',
    description='Quantum mechanical bespoke force field parameter generation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/qubekit/QUBEKit',
    packages=find_packages(),
    # install_requires=[
    #     'pyyaml',
    #     'py-cpuinfo',
    #     'psutil',
    #     'qcengine',
    #     'pydantic>=0.20.0',
    # ],
    author=['Joshua Thomas Horton', 'Chris Ringrose'],
    entry_points={
        'console_scripts': [
            'QUBEKit = QUBEKit.run:main',
            'qubekit = QUBEKit.run:main',
            'QUBEKit-pro = QUBEKit.protein_run:main'
        ]
    },
    version=datetime.now().strftime('%Y.%m.%d'),
    license='MIT',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Programming Language :: Python :: 3.6',
    ],
    python_requires='~=3.6'
)

print('Finding home directory to store QUBEKit config files')
home = str(Path.home())
config_folder = f'{home}/QUBEKit_configs/'

if not path.exists(config_folder):
    makedirs(config_folder)
    print(f'Making config folder at: {home}')
