import torch
import numpy as np
import os
from scipy.constants import physical_constants
from PYSEQM.seqm.seqm_functions.constants import Constants
from PYSEQM.seqm.basics import Energy
from PYSEQM import seqm
from Dispersion import DispersionCorrection

eV_to_Hartree = physical_constants['electron volt-hartree relationship'][0]
bohr_radius = physical_constants['Bohr radius'][0]
angstrom_per_meter = 1e10
bohr_to_angstrom = bohr_radius * angstrom_per_meter

# coordinates_ang in Angström
# Energy in eV and energy gradient in eV/Angström
def get_rate_and_gradient(coords, atomic_numbers, verbose=False, method='AM1', scf_eps = 1.0e-6, scf_converger = [2,0.0], sp2 =[True, 1.0e-5], pair_outer_cutoff = 1.0e10, cuda_device='cuda'):

    torch.set_default_dtype(torch.float64)
    if torch.cuda.is_available():
        device = torch.device(cuda_device)
    else:
        device = torch.device('cpu')
    
    atomic_numbers = np.array(atomic_numbers)
    # make sure the atomic number is sorted in descending order
    indx = (len(atomic_numbers) - 1 - np.argsort(atomic_numbers[::-1], kind='stable')[::-1]).astype(int)
    atomic_numbers = atomic_numbers[indx]

    coordinates_ang = np.copy(coords)
    coordinates_ang = coordinates_ang[indx]

    const = Constants().to(device)
    species = torch.tensor(atomic_numbers, dtype=torch.int64, device=device).unsqueeze(0)
    coordinates = torch.tensor(coordinates_ang, dtype=torch.float64, device=device).unsqueeze(0)
    coordinates.requires_grad_(True)
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

    with torch.autograd.set_detect_anomaly(True):
        eng = Energy(seqm_parameters).to(device)
        Hf, Etot, Eelec, Enuc, Eiso, EnucAB, e, P, charge, notconverged = eng(const, coordinates, species, all_terms=True)

        L=Etot[0]
        coordinates_grad=torch.autograd.grad(L,coordinates,create_graph=True,retain_graph=True)[0]

    energy = Etot.item()
    gradient = np.copy(coordinates_grad.cpu().detach().numpy())[0]
    #reorder to original order
    gradient_cp = np.copy(gradient)
    gradient[indx] = gradient_cp[:]

    #dispersion correction

    cartesian_coords = coords/bohr_to_angstrom

    atomslist = list(zip(atomic_numbers, cartesian_coords))
    diff = cartesian_coords[:, np.newaxis, :] - cartesian_coords[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    zero_distances = np.isclose(distances, 0.0)
    directions = np.zeros_like(diff)
    np.divide(diff, np.expand_dims(distances, axis=-1), where=~zero_distances[..., np.newaxis], out=directions)
    Nat = len(atomic_numbers)

    disp = DispersionCorrection(atomslist)
    energy = energy + disp.getEnergy(atomslist)/eV_to_Hartree
    gradient = gradient + disp.getGradient(atomslist,distances,directions).reshape((Nat, 3))/eV_to_Hartree/bohr_to_angstrom

    if verbose:
        print("Total Energy (eV): ", energy)
        print("Gradient: ")
        print(gradient)
    return (energy, gradient) #in eV and eV/Angström