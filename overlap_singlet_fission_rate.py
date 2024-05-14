#!/usr/bin/env python
"""
singlet fission rate according to Michl's model
"""
import torch

from PYSEQM.seqm.basics import Parser, Pack_Parameters, Hamiltonian
from PYSEQM.seqm.seqm_functions.diat_overlap import diatom_overlap_matrix
from PYSEQM.seqm.seqm_functions import constants
from PYSEQM.seqm.seqm_functions.diag import sym_eig_trunc1


BohrToAngstrom = 0.52917721067
    
def softmax(y, dim=0, beta=30.0):
    """
    differentiable substitute for argmax(y)
    
    max(y) = softmax(y)*y
    
    Example
    -------
    >>> y = torch.tensor([0.7, 0.99, 0.9, 0.1])
    >>> softmax(y)
    tensor([1.5607e-04, 9.3688e-01, 6.2963e-02, 2.3770e-12])
    >>> y * softmax(y)
    tensor([1.0925e-04, 9.2751e-01, 5.6667e-02, 2.3770e-13])
    """
    a = torch.exp(beta*y)
    b = torch.sum(a, dim).unsqueeze(dim)
    return a/b


class overlap_singlet_fission_rate(torch.nn.Module):
    def __init__(self, seqm_parameters, species, atom_indices_A, atom_indices_B, upper_device="cpu"):
        """
        computes |T_RP|^2 for a dimer

        Parameters
        ----------
        atom_indices_A :  Tensor (int,)
          indices of atoms belonging to monomer A
        atom_indices_B :  Tensor (int,)
          indices of atoms belonging to monomer B
        """
        super().__init__()
        self.device = upper_device
        
        self.const = constants.Constants().to(self.device)
        self.species = species.to(self.device)
        self.atom_indices_A = atom_indices_A.to(self.device)
        self.atom_indices_B = atom_indices_B.to(self.device)
        self.parser = Parser(seqm_parameters).to(self.device)
        self.packpar = Pack_Parameters(seqm_parameters).to(self.device)
        self.hamiltonian = Hamiltonian(seqm_parameters).to(self.device)

        # number of valence orbital for elements H,C,N and O
        self.num_valorbs = {1 : 1,
                            5 : 4,
                            6 : 4,
                            7 : 4,
                            8 : 4,
                            9 : 4}
        
    def forward(self, coordinates):
        """
        simple approximations for the singlet fission rate according to [1]

        References
        ----------
        [1] A. Buchanan et.al.
            "Singlet Fission: Optimization of Chromophore Dimer Geometry"
            http://dx.doi.org/10.1016/bs.aiq.2017.03.005
        """
        # MO coefficients of frontier orbitals
        hA,lA, _, _, _, _   = self.frontier_orbitals_subsystem(coordinates, self.atom_indices_A)
        hB,lB, _, _, _, _ = self.frontier_orbitals_subsystem(coordinates, self.atom_indices_B)
            
        # approximation according to eqn. (43) in [1]
        
        # AO overlap matrix between orbitals on fragments A and B, S_AB
        S_AB = self.overlap_AB(coordinates)
            
        nmol, naoA, naoB = S_AB.size()
    
        S = torch.zeros((nmol,2,2)).to(self.device)
        # Shh = S[0,0], Shl = S[0,1], Slh = S[1,0], Sll = S[1,1]

        for m in range(0, naoA):
            for n in range(0, naoB):
                S[...,0,0] += hA[...,m]*hB[...,n]*S_AB[...,m,n]
                S[...,0,1] += hA[...,m]*lB[...,n]*S_AB[...,m,n]
                S[...,1,0] += lA[...,m]*hB[...,n]*S_AB[...,m,n]
                S[...,1,1] += lA[...,m]*lB[...,n]*S_AB[...,m,n]
                
        t2 = (S[...,1,0]*S[...,1,1]-S[...,0,1]*S[...,0,0])**2

        return t2

        
    def overlap_AB(self, coordinates):
        """
        The AO overlap matrix for the combined system A+B has the following structure
        (after reordering atoms)
        
               S_AA  S_AB
           S =
               S_BA  S_BB
        
        We need to extract the block S_AB 
        """
        nmol, molsize, \
        nHeavy, nHydro, nocc, \
        Z, maskd, atom_molid, \
        mask, pair_molid, ni, nj, idxi, idxj, xij, rij = self.parser(self.const, self.species, coordinates)
        parameters = self.packpar(Z)
        zetas = parameters['zeta_s']
        zetap = parameters['zeta_p']
        zeta = torch.cat((zetas.unsqueeze(1), zetap.unsqueeze(1)),dim=1)
        di = diatom_overlap_matrix(ni, nj, xij, rij, zeta[idxi], zeta[idxj],
                                   self.const.qn_int)
        
        # number of unique atom pairs
        npair = (molsize*(molsize-1))//2
        
        # di contains the overlap matrix elements between unique atom pairs (i < j)
        di = di.reshape((nmol,npair,4,4))
    
        # expand the upper triangle in `di` to a full 2d overlap matrix `S`
        S = torch.zeros((nmol,molsize,molsize,4,4)).to(self.device)
        
        # atom indices of rows and columns 
        row,col = torch.triu_indices(molsize,molsize, offset=1)
        diag = torch.arange(molsize)
        # orbital indices for 4x4 subblocks
        col_orb  = torch.arange(4).unsqueeze(0).expand(4,4).reshape(-1)
        row_orb  = torch.arange(4).unsqueeze(1).expand(4,4).reshape(-1)
        # upper triangle of overlap matrix
        S[:,row,col  ,:, :] = di
        # diagonal of S, AOs are orthonormal
        S[:,diag,diag, 0,0] = 1.0
        S[:,diag,diag, 1,1] = 1.0
        S[:,diag,diag, 2,2] = 1.0
        S[:,diag,diag, 3,3] = 1.0
        # fill in lower triangle, S is symmetric
        for a in row_orb:
            for b in col_orb:
                S[:,col,row, b,a] = S[:,row,col, a,b]

        # lists of atom indices belonging to fragments A and B
        idxA, idxB = self.atom_indices_A, self.atom_indices_B
        
        # count the number of valence orbitals on fragments A and B
        naoA = sum([self.num_valorbs[int(element)] for element in self.species[0,idxA]])
        naoB = sum([self.num_valorbs[int(element)] for element in self.species[0,idxB]])
        # S_AB contains the overlap matrix elements between orbitals on fragment A
        # and fragment B.
        S_AB = torch.zeros((nmol,naoA,naoB)).to(self.device)

        # `m` enumerates orbitals on fragment A
        m = 0
        for i in range(0, molsize):
            # `n` enumerates orbitals on fragment B
            n = 0
            # number of valence orbitals on atom i (1 AO for H or 4 AOs for C,N,O)
            dm = self.num_valorbs[int(self.species[0,i])]
            for j in range(0, molsize):
                # number of valence orbitals on atom j
                dn = self.num_valorbs[int(self.species[0,j])]
                
                if (i in idxA) and (j in idxB):
                    S_AB[:,m:m+dm,n:n+dn] = S[:,i,j,0:dm,0:dn]
                
                if (j in idxB):
                    n += dn
                
            assert n == naoB
            if (i in idxA):
                m += dm
        assert m == naoA
    
        return S_AB
    
    
    def frontier_orbitals_subsystem(self, coordinates, atom_indices):
        """
        compute frontier orbitals for a subsystem
        """
        nmol, molsize, \
        nHeavy, nHydro, nocc, \
        Z, maskd, atom_molid, \
        mask, pair_molid, ni, nj, idxi, idxj, xij, rij = self.parser(self.const, self.species[:,atom_indices], coordinates[:,atom_indices,:])
        parameters = self.packpar(Z)
        F, e, P, Hcore, w, charge, notconverged = self.hamiltonian(self.const, molsize, nHeavy, nHydro, nocc, Z, maskd, mask, atom_molid, pair_molid, idxi, idxj, ni,nj,xij,rij, parameters)
        # e : orbital energies
        # v : MO coefficients (for all molecules and orbitals)
        e, v = sym_eig_trunc1(F,nHeavy, nHydro, nocc, eig_only=True)

        """
        ### DEBUG
        # save molecular orbitals
        orbs = torch.stack(v)
        orbe = e
        pyseqm_helpers.write_molden("test.molden", self.species[:,atom_indices], coordinates[:,atom_indices,:], orbs, orbe, nocc)
        ###
        """
        
        nao,nmo = v[0].shape
        
        # MO coefficients of HOMO and LUMO
        homo = torch.zeros((nmol, nao)).to(self.device)
        lumo = torch.zeros((nmol, nao)).to(self.device)
        lHl = torch.zeros((nmol)).to(self.device)
        hHh = torch.zeros((nmol)).to(self.device)
        for i in range(0, nmol):
            # If nocc is the number of occupied orbitals, nocc-1 should be the index of
            # the HOMO and nocc the index of the LUMO (using 0-based indices).
            nHOMO = nocc[i]-1
            nLUMO = nocc[i]
            homo[i,:] = v[i][:,nHOMO]
            lumo[i,:] = v[i][:,nLUMO]
            hHh[i] = e[i][nHOMO]
            lHl[i] = e[i][nLUMO] 
            
        return homo,lumo, nHeavy, nHydro, lHl, hHh 
