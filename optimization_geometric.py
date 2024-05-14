#!/usr/bin/env python

import singlet_fission_rate_and_gradient as sfr
import energy_rate_and_gradient_semi as energy
from scipy.constants import physical_constants
import numpy as np

from geometric.engine import Engine
from geometric.molecule import Molecule
from geometric.optimize import run_optimizer
from geometric.errors import GeomOptNotConvergedError
import tempfile
from utility import read_config, write_xyz, print_git_commit

import argparse

from ase.io import read

eV_to_Hartree = physical_constants['electron volt-hartree relationship'][0]
bohr_radius = physical_constants['Bohr radius'][0]
angstrom_per_meter = 1e10
bohr_to_angstrom = bohr_radius * angstrom_per_meter

class Singlet_Fission_Engine(Engine):

    def __init__(self, molecule, species, cuda_device, easy_delta_E, D_0, hardness, stability_correction, SF_down_correction, approximation, objective_function_name):
        super(Singlet_Fission_Engine, self).__init__(molecule)
        self.species = species
        self.cuda_device = cuda_device
        self.easy_delta_E = easy_delta_E
        self.D_0 = D_0
        self.hardness = hardness
        self.stability_correction = stability_correction
        self.SF_down_correction = SF_down_correction
        self.approximation = approximation
        if objective_function_name == 'activation':
            self.objective_function = activation_function
        else:
            raise NotImplementedError("This objective function is not implemented!")


    def calc_new(self, coords, dirname):
        # coords in Bohr
        coords = coords*bohr_to_angstrom # in Angström
        global counter
        sf_rate, sfr_grad_linear, E, E_grad_linear, coords = calculate(coords, self.species, self.cuda_device, self.easy_delta_E, self.approximation)
        score, score_grad = self.objective_function(sf_rate, sfr_grad_linear, E, E_grad_linear, self.species, self.D_0, hardness = self.hardness, stability_correction = self.stability_correction, SF_down_correction = self.SF_down_correction)
        printAndSave(E, sf_rate, score, coords, self.species, self.D_0, name=f"{counter}: ")

        score_grad = score_grad*bohr_to_angstrom*eV_to_Hartree # in E_h/Bohr
        score = score*eV_to_Hartree # in E_h

        return {'energy': score, 'gradient': score_grad}


counter = 0

def calculate(linear_coords, species, cuda_device, easy_delta_E, approximation):
    nmol = len(species)
    global counter

    linear_coords = np.array(linear_coords)
    coords = linear_coords.reshape((nmol,3))
    

    (sf_rate, sfr_grad)  = sfr.get_rate_and_gradient(coords, species, cuda_device = cuda_device, resultPrintOut = False, easy_delta_E = easy_delta_E, approximation = approximation)
    (E, E_grad)  = energy.get_rate_and_gradient(coords, species, cuda_device = cuda_device)

    sfr_grad = np.array(sfr_grad)
    sfr_grad_linear = sfr_grad.reshape((nmol*3))
    E_grad = np.array(E_grad)
    E_grad_linear = E_grad.reshape((nmol*3))
    
    counter += 1

    return sf_rate, sfr_grad_linear, E, E_grad_linear, coords

def activation_function(sf_rate, sfr_grad_linear, E, E_grad_linear, species, D_0, hardness, stability_correction, SF_down_correction):

    score = E - hardness * np.log(sf_rate)
    score_grad = E_grad_linear - hardness * sfr_grad_linear / sf_rate

    return score, score_grad

def printAndSave(E, sf_rate, score, coords, species, D_0, name):
    print(f"{name}")
    print(f"Energy: {E:.5f} eV")
    print(f"ΔE: {E-D_0:.5f} eV")
    print(f"SFR: {sf_rate:.5e}")
    print(f"Score: {score:.5e}")
    print('*'*50)
    write_xyz(coords, species, 'traj.xyz', comment = f'E: {E:.5f} eV; SFR: {sf_rate:.5e}; score: {score:.5e}', mode="a")



def main(dimer_file, cuda_device = 'cpu', easy_delta_E = True, hardness = 0.5, stability_correction = 0., SF_down_correction = 1e-7, maxiter = 200, structure_name = 'struct', approximation = 'overlap', objective_function = 'activation'):

    all_atoms = read(dimer_file, format='xyz',index=":")[0]
    coords = all_atoms.get_positions()
    coords = np.array(coords)
    species = all_atoms.get_atomic_numbers()
    species = np.array(species)

    nmol = len(species)

    (E_monomer, _)  = energy.get_rate_and_gradient(coords[:nmol//2,:], species[:nmol//2], cuda_device = cuda_device)

    D_0 = E_monomer*2

    print(f'D_0: {D_0:.5f} eV')

    mole = Molecule(dimer_file, build_topology = True)
    engine = Singlet_Fission_Engine(mole, species, cuda_device, easy_delta_E, D_0, hardness, stability_correction, SF_down_correction, approximation, objective_function)
    tmpf = tempfile.mktemp()

    try:
        m = run_optimizer(customengine=engine, check=0, input=tmpf, maxiter = maxiter, coordsys='dlc', conmethod=1, verbose=3, logIni="log.ini",prefix=structure_name)
        m.write(fnm='opt_traj.xyz', ftype='xyz')
        print('Optimisation converged!')
    except GeomOptNotConvergedError:
        print('Optimisation not converged!')

if __name__ == '__main__':

    print_git_commit()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help='path to the json config file', default="config.json")

    args = parser.parse_args()


    config_file = args.config

    config: dict = read_config(config_file)


    main(config.get('structure', 'dimer.xyz'), cuda_device = config.get('cuda_device', 'cpu'), easy_delta_E = config.get('easy_delta_E', True), hardness=config.get('hardness', 0.5),
          stability_correction = config.get('stability_correction', 0.0), SF_down_correction=config.get('SF_down_correction', 1e-7), maxiter = config.get('maxiter', 200),
        structure_name = config.get('structure_name', 'struct'), approximation = config.get('approximation', 'overlap'), objective_function = config.get('objective_function', 'activation'))

    print('Optimisation program finished!')
    