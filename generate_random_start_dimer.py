#!/usr/bin/env python

from ase.io import read
from ase.io.extxyz import write_extxyz
from ase.atoms import Atoms
import numpy as np
import random
import linecache
import argparse

def make_random_Dimer(input_structure, dimer_structure, iterations = 200, minimal_dist = 2., maximal_dist = 3.):
    monomer = read(input_structure)
    comment = linecache.getline(input_structure, 2)

    monomer_coords = np.array(monomer._get_positions())
    monomer_species = monomer.get_chemical_symbols()
    monomer_nat = len(monomer_species)

    dimer_coords = np.zeros((monomer_nat*2,3))
    dimer_species = []

    min_dist = 0.5
    range_ = max(np.max(monomer_coords) - np.min(monomer_coords),min_dist+0.1)

    success = False

    for itera in range(iterations):

        dimer_coords = np.zeros((monomer_nat*2,3))
        dimer_species = []

        dx = random.uniform(min_dist,range_)
        dy = random.uniform(min_dist,range_)
        dz = random.uniform(min_dist,range_)
        dx_rot = random.uniform(0,2*np.pi)
        dy_rot = random.uniform(0,2*np.pi)
        dz_rot = random.uniform(0,2*np.pi)

        for i in range(monomer_nat):
            dimer_species.append(monomer_species[i])
            dimer_coords[i,:] = monomer_coords[i,:]
        for i in range(monomer_nat):
            dimer_species.append(monomer_species[i])
            x,y,z = monomer_coords[i,:]
            x_n = x*np.cos(dy_rot)*np.cos(dz_rot) - y*np.sin(dz_rot)*np.cos(dy_rot) + z*np.sin(dy_rot)
            y_n = x*(np.sin(dx_rot)*np.sin(dy_rot)*np.cos(dz_rot) + np.sin(dz_rot)*np.cos(dx_rot)) + y*(-np.sin(dx_rot)*np.sin(dy_rot)*np.sin(dz_rot) + np.cos(dx_rot)*np.cos(dz_rot)) - z*np.sin(dx_rot)*np.cos(dy_rot)
            z_n = x*(np.sin(dx_rot)*np.sin(dz_rot) - np.sin(dy_rot)*np.cos(dx_rot)*np.cos(dz_rot)) + y*(np.sin(dx_rot)*np.cos(dz_rot) + np.sin(dy_rot)*np.sin(dz_rot)*np.cos(dx_rot)) + z*np.cos(dx_rot)*np.cos(dy_rot)
            dimer_coords[i+monomer_nat,:] = np.array([x_n+dx,y_n+dy,z_n+dz])

        dist = np.zeros((monomer_nat,monomer_nat,3))
        for i in range(monomer_nat):
            dist[i,:,:] = dimer_coords[:monomer_nat,:] - np.roll(dimer_coords[monomer_nat:,:], i, axis=0)
        dist = np.square(dist)
        dist = np.sum(dist,axis=2)
        shortest_distance = np.sqrt(np.min(dist))
        if shortest_distance > minimal_dist and shortest_distance < maximal_dist:
            success = True
            break

    print(f'Iterations: {itera+1}')

    if success:
        dimer = Atoms(dimer_species, dimer_coords)
        print(dimer)
        write_extxyz(dimer_structure, dimer, comment=f'Dimer: {comment}')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                    type=str,
                    help='XYZ file with geometry of monomer used to build the random dimer')
    parser.add_argument('output',
                    type=str,
                    help='XYZ file with geometry of monomer used to build the random dimer')
    args = parser.parse_args()

    make_random_Dimer(args.input, args.output)
    