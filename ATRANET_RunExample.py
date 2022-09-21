# -*- coding: utf-8 -*-
"""
Example Script for using the ATRANET_v2 code to calculated Transition Networks from MD data.

This Script needs you to specify:

    top:    structure file in .gro/.pdb format

    traj:   system traj in .xtc/.trr format

    TMname: nameing convention for output files

    state:  descriptors used to calculate the network. This version contains:
                - residueInHelix
                - residueInBeta
                - residueInCoil
                - EndToEnd
                - Rg
                - OrderParameter
                - ProtProtContacts
                - ProtProtContactsChain
                - ProtLigContacts
                - OligomericSize
                - hydrophobicContacts
                - polarContacts
                - intermolecularSaltBridge
                - Ramachandran

    desc:   dictionary containing distance used for:
                - ProtProtContacts
                - ProtProtContactsChain
                - ProtLigContacts
                - OligomericSize
                - hydrophobicContacts
                - polarContacts
                - intermolecularSaltBridge

    nProt:    Number of protein chains

    ResProt:  Define length of the main protein ie. number of residual elements (first in .gro/.pdb file)

    nLig:     Number of Ligand chains

    ResLig:   Define length of the ligand ie. number of residual elements / building blocks (second in .gro/.pdb)

Optional keyargs:

    res:      Defines the binning width of the Rg/EndToEnd routine in Angstrom

    traj_suf: specifies the suffix of the input trajectories (default='.xtc')
"""

from ATRANET import TransitionNetworks
import numpy as np

#################################################################################################
########################################## Input Data ###########################################
#################################################################################################

TMname = 'TM_SystemName'
top = '/path/to/system.pdb'
traj = '/path/to/trajectory*ies/system_prefix'

#################################################################################################
################################## Define Descriptor Functions ##################################
#################################################################################################

""" We are considering a hexamer simulation of short proteins with 14 residues each """
nProt = 6
ResProt = 14
nLig = 0
ResLig = 0


"""
Define Descriptors:
    1. Calculate the number of inter protein contacts between residues of different protein chains where a distance below 10 Angsrom is considered a contact.
    2. Calculate size of the largest Oligomer, considering two protein chains closer than 10 Angstrom to be in contact.
    3. Number of residues in alpha structure
    4. Number of residue in beta sheet structure
"""
desc = {'ProtProtContacts': 10, 'OligomericSize': 10}
state = ['ProtProtContacts', 'OligomericSize' ,'residuesInHelix', 'residuesInBeta']


#################################################################################################
################################ Initialize and Generate Network ################################
#################################################################################################

""" Initialize """
tn = TransitionNetworks(top=top, traj=traj,
                        nProt=nProt , ResProt=ResProt, nLig=nLig, ResLig=ResLig,
                        desc=desc, state=state, res=2)



""" Running generateNetwork() will calculate the TM and output the Graph as .gefx """
tn.GenerateNetwork(gexfName=TMname+'.gexf',TransitionMatrixName=TMname+'.npy', DictionaryName=TMname+'Dict.txt', statetrjName=TMname+'state_trj.txt')


""" Print correlation between descriptors to the terminal """
tn.CorrelationCoefficients(trjpath=TMname+'state_trj.txt')


























