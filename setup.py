from setuptools import setup

setup(
    name='qubekit',
    description='Quantum mechanical bespoke force field parameter generation',
    url='https://github.com/cole-group/QuBeKit',
    author='Joshua Thomas Horton',
    packages=['QUBEKit'],
    python_requires='>=3',
    entry_points={'console_scripts': [
        'QUBEKit = QUBEKit.run:main',
    ],},
    install_requires=[
        'numpy>=1.11'
    ])

