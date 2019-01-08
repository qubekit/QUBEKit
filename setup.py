# !/usr/bin/env python


from setuptools import setup, find_packages
from pathlib import Path
from os import path, makedirs

setup(
    name='qubekit',
    description='Quantum mechanical bespoke force field parameter generation',
    url='https://github.com/jthorton/QUBEKitdev',
    packages=find_packages(),
    author=['Joshua Thomas Horton', 'Chris Ringrose'],
    entry_points={'console_scripts': [
        'QUBEKit = QUBEKit.run:Main',
        'QUBEKit-josh = QUBEKit.tests_josh:main'
    ],},
    )

print('Finding home directory to store QUBEKit config files')
home = str(Path.home())
config_folder = f'{home}/QUBEKit_configs/'
if not path.exists(config_folder):
    makedirs(config_folder)
    print(f'Making config folder at: {home}')
