#!/usr/bin/env python

import torch
import os.path
import numpy as np
from PYSEQM import seqm
from overlap_singlet_fission_rate import overlap_singlet_fission_rate as SFR_overlap
from scipy.constants import physical_constants

eV_to_Hartree = physical_constants['electron volt-hartree relationship'][0]
bohr_radius = physical_constants['Bohr radius'][0]
angstrom_per_meter = 1e10
bohr_to_angstrom = bohr_radius * angstrom_per_meter

# coordinates_ang in Angström
# Energy in eV and energy gradient in eV/Angström

def get_rate_and_gradient(coordinates_ang, atomic_numbers, verbose=False, method='AM1', scf_eps = 1.0e-6, scf_converger = [2,0.0], sp2 =[True, 1.0e-5], pair_outer_cutoff = 1.0e10, cuda_device = 'cuda', resultPrintOut=True, two_electron_integrals = True, easy_delta_E = False, approximation = "overlap"):
    torch.set_default_dtype(torch.float64)
    if torch.cuda.is_available():
        device = torch.device(cuda_device)
    else:
        device = torch.device('cpu')

    if not approximation in ["overlap"]:
        raise NotImplementedError("The given approximation is not implemented!")

    atomlist = list(zip(atomic_numbers, coordinates_ang))
    dimer = atomlist
    molsize = len(dimer)

    sort_idx = torch.argsort(torch.tensor([Zat for (Zat,pos) in dimer]), descending=True)
    # reverse mapping from sorted indices to original ordering
    unsort_idx = torch.zeros(molsize, dtype=torch.int64)
    for i in range(0, molsize):
        unsort_idx[sort_idx[i]] = i

    # Now the atoms are sorted by atomic number
    dimer = [dimer[i] for i in sort_idx] 

    assert molsize % 2 == 0

    atom_indices_A = []
    atom_indices_B = []
    for i in range(0, molsize):
        if i in unsort_idx[:molsize//2]:
            atom_indices_A.append(i)
        elif i in unsort_idx[molsize//2:]:
            atom_indices_B.append(i)

    atom_indices_A = torch.tensor(atom_indices_A)
    atom_indices_B = torch.tensor(atom_indices_B)
    if verbose:
        print("indices of atoms of fragment A: %s" % atom_indices_A)
        print("indices of atoms of fragment B: %s" % atom_indices_B)
        print("atomic numbers: %s" % [atom[0] for atom in dimer])

    species = torch.ones((1,molsize),dtype=torch.int64)
    coordinates = torch.zeros((1,molsize,3))

    for i, d in enumerate(dimer):
        Zat,(x,y,z) = d
        species[0,i] = Zat
        coordinates[0,i,:] = torch.tensor([x,y,z])

    if verbose:
        print("atomic numbers of fragment A: ", [species[0,i].item() for i in atom_indices_A])
        print("atomic numbers of fragment B: ", [species[0,i].item() for i in atom_indices_B])

    elements = [0]+sorted(set(species.reshape(-1).tolist()))
    seqm_parameters = {
                    'method' : method,  # AM1, MNDO, PM#
                    'scf_eps' : scf_eps,  # unit eV, change of electric energy, as nuclear energy doesnt' change during SCF
                    'scf_converger' : scf_converger, # converger used for scf loop
                                            # [0, 0.1], [0, alpha] constant mixing, P = alpha*P + (1.0-alpha)*Pnew
                                            # [1], adaptive mixing
                                            # [2], adaptive mixing, then pulay
                    'sp2' : sp2,  # whether to use sp2 algorithm in scf loop,
                                                #[True, eps] or [False], eps for SP2 conve criteria
                    'elements' : elements, #[0,1,6,8],
                    'learned' : [], # learned parameters name list, e.g ['U_ss']
                    'parameter_file_dir' : os.path.normpath(os.path.join(os.path.dirname(seqm.__file__),"../params/MOPAC"))+"/",
                    'pair_outer_cutoff' : pair_outer_cutoff, # consistent with the unit on coordinates
                    }

    if verbose:
        print(f"Calculation will be run on device '{device}'")

    with torch.autograd.set_detect_anomaly(True):    
        coordinates = coordinates.to(device)
        coordinates.requires_grad_(True)
        if approximation == "overlap":
            sfr = SFR_overlap(seqm_parameters, species,
                                    atom_indices_A, atom_indices_B,
                                    upper_device=cuda_device).to(device)
        else:
            print("Error approximation not found!")
            exit()

        # compute rates
        t2 = sfr(coordinates)           
        coordinates_grad = torch.autograd.grad(t2,coordinates)
    
    fissionRate = t2.item() #eV
    fissionRate_grad = np.copy(coordinates_grad[0].cpu().detach().numpy())[0] #eV/Angström

    #reorder gradients to original order
    fissionRate_grad = fissionRate_grad[unsort_idx]
    
    if verbose:
        print("Singlett Fission Rate: ",fissionRate)
        print("Singlett Fission Gradient: ")
        print(fissionRate_grad)
        print(f"reordered: {species[0,unsort_idx]}")
    
    cv = fissionRate
    cv_grad = fissionRate_grad

    if resultPrintOut:
        print("Singlet Fission Rate: ", fissionRate)

    return (cv, cv_grad)
