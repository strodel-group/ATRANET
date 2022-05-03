# -*- coding: utf-8 -*-
"""
Example Script for using the ATRANET_v2 code to calculated Transition Networks from MD data.

This Script needs you to specify:

top:    structure file in .gro/.pdb format
traj:   system traj in .xtc/.trr format
TMname: name of most output files
desc:   dictionary containing distance used for 'allContacts'
state:  descriptors used to calculate the network. This version contains:
            - residueInHelix
            - residueInBeta
            - residueInCoil
            - allContacts
            - EndToEnd
            - Rg
nProt:  Define length of the main protein (first in .gro/.pdb file)
nLig:   Define length of the ligand (second in .gro/.pdb)

This Script Outputs:
TMname.gexf:    Graph file which can be read in with gephi
TMname.npy:     numpy file containing the Transition Matrix
TMnameDict.txt: Dictionary of TN states
state_trj.txt:  trajectory of states (corralates traj. frames with states)
"""

from ATRANET_v2 import TransitionNetworks

""" Input Data """
TMname = 'TM_SystemName'
top = '/path/to/system.pdb'
traj = '/path/to/trajectory*ies/system_prefix'

""" Define Descriptor Functions """

desc = {'allContacts': 10.0}
state = ['allContacts', 'residuesInHelix', 'residuesInBeta', 'EndToEnd']

""" Define length of the main protein (first in .gro/.pdb file) and ligand (second in .gro/.pdb)
    In case of a monomer set nLig=0 or for dimers nLig = nProt """

nProt = 42
nLig = 16

""" Initialize Transition Network Class 
    res: resolution or binwidth for EtE/Rg """

tn = TransitionNetworks(top=top, traj=traj, desc=desc,
                        nProt=nProt ,nLig=nLig, state=state, res=2)


""" Running generateNetwork() will calculate the TM and output the Graph as .gefx """

tn.generateNetwork(gexfName=TMname+'.gexf',TransitionMatrixName=TMname+'.npy', DictionaryName=TMname+'Dict.txt')




