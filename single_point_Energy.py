#!/usr/bin/env python

import energy_rate_and_gradient_semi as e_rate
import numpy as np

from utility import read_config, print_git_commit
import argparse
from ase.io import read

def calculate(coords, species, cuda_device):
    (energy, energy_grad)  = e_rate.get_rate_and_gradient(coords, species, cuda_device = cuda_device)
    return energy, energy_grad


def printAndSave(energy, energy_grad, save_name):
    print(f"Energy: {energy:.5e} eV")
    print("Gradient:")
    for i in range(len(energy_grad)):
        buf = ""
        for j in range(len(energy_grad[0])):
            buf+= f"{energy_grad[i,j]:20.5e}     "
        print(buf)
    np.savez(save_name, energy = energy, e_grad = energy_grad)

def main(dimer_file, cuda_device = 'cpu', save_name = 'sfr_and_grad'):

    all_atoms = read(dimer_file, format='xyz',index=":")[0]
    coords = all_atoms.get_positions()
    coords = np.array(coords)
    species = all_atoms.get_atomic_numbers()
    species = np.array(species)

    energy, energy_grad = calculate(coords, species, cuda_device)
    printAndSave(energy, energy_grad, save_name)

if __name__ == '__main__':

    print_git_commit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help='path to the json config file', default="config.json")

    args = parser.parse_args()
    config_file = args.config
    config: dict = read_config(config_file)
    main(config.get('structure', 'dimer.xyz'), cuda_device = config.get('cuda_device', 'cpu'), save_name = config.get('save_name', 'energy_and_grad'))
    print('Single point program finished!')
    