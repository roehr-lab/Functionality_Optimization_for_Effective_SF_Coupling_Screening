#!/usr/bin/env python

import energy_rate_and_gradient_semi as energy
from scipy.constants import physical_constants
import numpy as np
import os

from geometric.engine import Engine
from geometric.molecule import Molecule
from geometric.optimize import run_optimizer
from geometric.errors import GeomOptNotConvergedError
import tempfile

import argparse

from utility import write_xyz, create_atoms_from_xyz, print_git_commit


eV_to_Hartree = physical_constants['electron volt-hartree relationship'][0]
bohr_radius = physical_constants['Bohr radius'][0]
angstrom_per_meter = 1e10
bohr_to_angstrom = bohr_radius * angstrom_per_meter

counter = 0
coords_hist = []
energy_hist = []

class PYSEQM_Engine(Engine):

    def __init__(self, molecule, species, cuda_device, E_start):
        super(PYSEQM_Engine, self).__init__(molecule)
        self.species = species
        self.cuda_device = cuda_device
        self.E_start = E_start

    def calc_new(self, coords, dirname):
        # coords in Bohr
        coords = coords*bohr_to_angstrom # in Angström
        nmol = len(self.species)
        coords = coords.reshape((nmol,3))
        global counter
        (E, E_grad)  = energy.get_rate_and_gradient(coords, self.species, cuda_device = self.cuda_device)
        printAndSave(E, coords, self.species, self.E_start, name=f"{counter}: ")
        counter += 1

        
        E_grad = np.array(E_grad)
        E_grad_linear = E_grad.reshape((nmol*3))
        E_grad_linear = E_grad_linear*bohr_to_angstrom*eV_to_Hartree # in E_h/Bohr
        E = E*eV_to_Hartree # in E_h

        return {'energy': E, 'gradient': E_grad_linear}


def printAndSave(E, coords, species, E_start, name):
    global coords_hist
    global energy_hist
    print(f"{name}")
    print(f"Energy: {E:.5f} eV")
    print(f"ΔE: {E-E_start:.5f} eV")
    print('*'*50)
    write_xyz(coords, species, 'traj.xyz', comment = f'E: {E:.5f} eV; ΔE: {E-E_start:.5f} eV', mode="a")
    energy_hist.append(E)
    coords_hist.append(np.copy(np.array(coords)))

def main(file_name, cuda_device, maxiter = 200):

    structure_name = os.path.splitext(os.path.basename(file_name))[0]

    (all_atoms, _) = create_atoms_from_xyz(file_name)[0]
    coords = all_atoms.get_positions()
    coords = np.array(coords)
    species = all_atoms.get_atomic_numbers()
    species = np.array(species)

    (E_start, _)  = energy.get_rate_and_gradient(coords, species, cuda_device = cuda_device)

    mole = Molecule(file_name, build_topology = True)
    engine = PYSEQM_Engine(mole, species, cuda_device, E_start)
    # /home/kavkan/STILBENE/surface/2_cis_3060_dihedral

    tmpf = tempfile.mktemp()

    try:
        m = run_optimizer(customengine=engine, check=0, input=tmpf, maxiter = maxiter, coordsys='dlc', conmethod=1, verbose=3, logIni="log.ini",prefix=structure_name)
        m.write(fnm='opt_traj.xyz', ftype='xyz')
        print('Optimisation converged!')
    except GeomOptNotConvergedError:
        print('Optimisation not converged!')
    
    global coords_hist
    global energy_hist
    if len(energy_hist) > 0:
        energies = np.array(energy_hist)
        opt_ind = np.argmin(energies)
        write_xyz(coords_hist[opt_ind], species, filename=f"{structure_name}_optimal.xyz", comment=f"{structure_name}; Energy: {energies[opt_ind]:.5f} eV")


if __name__ == '__main__':

    print_git_commit()

    parser = argparse.ArgumentParser()
    parser.add_argument("--structure", type=str, help='path to the xyz file')
    parser.add_argument("--cuda_device", type=str, help='name of the cuda device', default='cpu')
    parser.add_argument("--maxiter", type=str, help='maximal interations in the optimization', default=200)
    args = parser.parse_args()

    main(args.structure, cuda_device = args.cuda_device)
    print('Optimization program finished!')
    