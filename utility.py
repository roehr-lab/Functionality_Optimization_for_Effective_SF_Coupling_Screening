#!/usr/bin/env python

import json
import os
import periodictable
import subprocess
from ase import Atoms
    
def write_xyz(coordinates, species, filename, comment = "auto-generated", mode="w"):
    nat = len(coordinates)
    assert nat == len(species)
    with open(filename, mode) as f:
        f.write(f"{nat}\n")
        f.write(f"{comment.strip()}\n")
        for c, s in zip(coordinates, species):
            f.write(f"{str(periodictable.elements[s])}\t{c[0]:.6f}\t{c[1]:.6f}\t{c[2]:.6f}\n")

def create_atoms_from_xyz(filename):
    structures = []
    index = 0
    with open(filename, 'r') as file:
        lines = file.readlines()
        while(index+1 < len(lines)):
            if lines[index].strip() == '':
                break
            nat = int(lines[index])
            comment = lines[index+1].strip()
            index += 2
            try:
                symbols = [str(periodictable.elements[int(line.split()[0])]) for line in lines[index:index+nat]]
            except:
                symbols = [line.split()[0] for line in lines[2:]]
            coordinates = [[float(coord) for coord in line.split()[1:]] for line in lines[index:index+nat]]
            structures.append((Atoms(symbols=symbols, positions=coordinates),comment))
            index += nat

        return structures
    
def execute_command(command):
    '''execute command (string) in shell and return output (string)'''
    # Start the process and capture the output
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
         # Check the return code of the process
    if result.returncode != 0:
        print("Process has finished with a non-zero exit code:", result.returncode)
    # Convert output to strings
    output = result.stdout
    return output

def print_git_commit():
    path = os.path.dirname(os.path.abspath(__file__))
    commit_id = execute_command(f'cd {path}; git log -1 --pretty=format:"%H"')
    print(f'Git commit: {commit_id}')

def read_config(config_file):
    if os.path.splitext(config_file)[1] != '.json':
        raise ValueError('File is not a json file.')
    
    with open(config_file) as f:
        data = json.load(f)

    return data