#!/usr/bin/env python

import singlet_fission_rate_and_gradient as sfr
import numpy as np

from utility import read_config, print_git_commit
import argparse
from ase.io import read

def calculate(coords, species, cuda_device, easy_delta_E, approximation):
    (sf_rate, sfr_grad)  = sfr.get_rate_and_gradient(coords, species, cuda_device = cuda_device, resultPrintOut = False, easy_delta_E = easy_delta_E, approximation = approximation)
    return sf_rate, sfr_grad


def printAndSave(sf_rate, sfr_grad, save_name):
    print(f"SFR: {sf_rate:.5e}")
    print("Gradient:")
    for i in range(len(sfr_grad)):
        buf = ""
        for j in range(len(sfr_grad[0])):
            buf+= f"{sfr_grad[i,j]:20.5e}     "
        print(buf)
    np.savez(save_name, sf_rate = sf_rate, sfr_grad = sfr_grad)

def main(dimer_file, cuda_device = 'cpu', easy_delta_E = True, approximation = 'overlap', save_name = 'sfr_and_grad'):

    all_atoms = read(dimer_file, format='xyz',index=":")[0]
    coords = all_atoms.get_positions()
    coords = np.array(coords)
    species = all_atoms.get_atomic_numbers()
    species = np.array(species)

    sf_rate, sfr_grad = calculate(coords,species, cuda_device, easy_delta_E, approximation)
    printAndSave(sf_rate, sfr_grad, save_name)

if __name__ == '__main__':

    print_git_commit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help='path to the json config file', default="config.json")

    args = parser.parse_args()
    config_file = args.config
    config: dict = read_config(config_file)
    main(config.get('structure', 'dimer.xyz'), cuda_device = config.get('cuda_device', 'cpu'), easy_delta_E = config.get('easy_delta_E', True), approximation = config.get('approximation', 'overlap'), save_name = config.get('save_name', 'sfr_and_grad'))
    print('Single point program finished!')
    