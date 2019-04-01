# !/usr/bin/env python


from setuptools import setup, find_packages
from pathlib import Path
from os import path, makedirs


here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as file:
    long_description = file.read()

setup(name='qubekit',
      description='Quantum mechanical bespoke force field parameter generation',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/jthorton/QUBEKitdev',
      packages=find_packages(),
      author=['Joshua Thomas Horton', 'Chris Ringrose'],
      entry_points={'console_scripts': [
          'QUBEKit = QUBEKit.run:main',
          'qubekit = QUBEKit.run:main',
          'QUBEKit-josh = QUBEKit.tests_josh:main',
          'QUBEKit-chris = QUBEKit.tests_chris:main',
          'QUBEKit-pro = QUBEKit.protien_run:main']},
      version='2.0.0',
      license='MIT',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
          'Topic :: Scientific/Engineering :: Chemistry',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
      ],
      python_requires='~=3.5',
      install_requires=[
          'matplotlib',
          'networkx',
          'numpy',
          'scipy',
      ])

print('Finding home directory to store QUBEKit config files')
home = str(Path.home())
config_folder = f'{home}/QUBEKit_configs/'

if not path.exists(config_folder):
    makedirs(config_folder)
    print(f'Making config folder at: {home}')
