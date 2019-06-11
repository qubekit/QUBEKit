import argparse
import os
import shutil
import subprocess as sp

parser = argparse.ArgumentParser(description='Creates a conda environment from file for a given Python version.')
parser.add_argument('-n', '--name', type=str, nargs=1, help='The name of the created Python environment')
parser.add_argument('-p', '--python', type=str, nargs=1, help='The version of the created Python environment')
parser.add_argument('conda_file', nargs='*', help='The file for the created Python environment')

args = parser.parse_args()

script = open(args.conda_file[0]).read()

tmp_file = 'tmp_env.yaml'
script = script.replace('- python', f'- python {args.python[0]}*')

with open(tmp_file, 'w') as handle:
    handle.write(script)

conda_path = shutil.which('conda')

print(f'CONDA ENV NAME  {args.name[0]}')
print(f'PYTHON VERSION  {args.python[0]}')
print(f'CONDA FILE NAME {args.conda_file[0]}')
print(f'CONDA path      {conda_path}')

sp.call(f'{conda_path} env create -n {args.name[0]} -f {tmp_file}', shell=True)
os.unlink(tmp_file)
