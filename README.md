# Automated-Transition-Networks (ATRANET)
A Python script that builds Automated-Transition-Networks for aggregating proteins, based on a set of descriptor functions.
The script is designed in a way that it allows the user to easily add further descriptor functions if needed, that is if they seem to be beneficial for the system under investigation.
To perform a TN analysis using ATRANET, the topology and trajectory files of all-atom MD simulations containing any number of polypeptide chains have to be provided.
In its current version, these input files have to be processable by MDAnalysis as well as mdtraj and by default are set to in Gromacs format .gro and .xtc, in order to use other input formats make use of the "traj_suf" keyarg of he initializer.
The ATRANET script invokes the MD trajectory analysis software MDTraj and MDAnalysis to calculate the descriptors specified by the user.
It then produces a file that contains the transition matrix, including the states' populations and the number of transitions between the states.
The transition matrix is saved as a .gexf file, which can be visualized by the network visualization software Gephi.

The script was originally written by Alexander-Maurice Illig (see JCTC-2020 branch https://github.com/strodel-group/ATRANET/tree/JCTC-2020-https/doi.org/10.1021/acs.jctc.0c00727) and later rewritten by Moritz Schäffler (master branch). Additional contributers are Suman Samantray and Mohammed Khaled.

For an extended explanation on the usage of Transition Networks and the provided descriptor functions, please refer to:
https://www.sciencedirect.com/science/article/abs/pii/S1046202322001670.

# ATRANET.py
The ATRANET.py file contains the main TransitionNetworks class that carries out all the calculations and utilizes the following python packages:
MDAnalysis, mdtraj, sklearn, numba, warnings, networkx, numpy, glob, sys, ast, pathlib

Most of these packages should be installed by default, otherwise the user has to install them manually.

The ATRANET.py script itself can be imported in any python script, in order to make its functions available:

    from ATRANET import TransitionNetworks

The class is then initialized by calling:

    tn = TransitionNetworks(*keyargs)

## Key Arguments
The TransitionNetworks class needs you to specify the following keyargs:

    top:    structure file in .gro/.pdb format

    traj:   system traj in .xtc/.trr format

    TMname: nameing convention for output files

    state:  descriptors used to calculate the network. This version contains:
                - residueInHelix
                - residueInBeta
                - residueInCoil
                - EndToEnd
                - Rg
                - CompactnessFactor
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

## Functions
The class contains the following functions:

### tn.GenerateTransitionMatrix()

Based on the choosen keyargs choosen when initializing the class, this function calculates the state of each frame of a provided Trajectory and constructs the Transition Matrix. 

Parameters:
* writetrajectory: \<bool\>

  writes the trajectory of states to an output .txt file. This can be usufull for correlating states and frames of the input trajectorym aswell as caclculating the correlation coefficient between descritptors.
  
  default = True

### tn.GenerateNetwork()

  Constructs the Transition Network from a Transition Matrix and dictionary of states. 
  
  The output network file shows the states ( nodes )  size as the population of the states.


# ATRANET_RunExample.py

The ATRANET_RunExample.py demonstrate the usage of the ATRANET.py script.

