# !/usr/bin/env python


from setuptools import setup, find_packages

setup(
    name='qubekit',
    description='Quantum mechanical bespoke force field parameter generation',
    url='https://github.com/cole-group/QuBeKit',
    author='Joshua Thomas Horton',
    packages=['QUBEKit'],
    entry_points={'console_scripts': [
        'QUBEKit = QUBEKit.run:main',
    ],},
    )
