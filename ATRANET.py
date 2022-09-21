# -*- coding: utf-8 -*-
"""
ATRANET (Automated TRAnsition NETworks) code for caclulating TNs from MD data.
Originaly written by:
Improved and modified by: Moritz Schäffler

The presented ATRANET code can be used to calculate TN for a dimer, protein-ligand or monomer system.

The script is designed in a way that it allows the user easily to add further
descriptor functions to the script if they seem to be more beneficial for the system under
investigation. To perform the TNs analysis, topology and trajectory files of
all-atom MD simulations with any number of peptide chains have to be provided.
From this, the script will produce a file that contains the transition
matrix and the state’s populations (”size”), which can be visualized
by network visualization software like Gephi.

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

This Script Outputs:
TMname.gexf:    Graph file which can be read in with gephi
TMname.npy:     numpy file containing the Transition Matrix
TMnameDict.txt: Dictionary of TN states
state_trj.txt:  trajectory of states (corralates traj. frames with states)
"""

############################################################################################################
######################################          Modules          ###########################################
############################################################################################################

import MDAnalysis as md # distances in Angstroem
from MDAnalysis.lib.pkdtree import PeriodicKDTree
from MDAnalysis.lib.distances import distance_array
from MDAnalysis.analysis.dihedrals import Ramachandran
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA

from sklearn.cluster import KMeans

from numba import jit
from numba import prange
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import mdtraj # necessary for dssp

import networkx as nx
import numpy as np
import h5py as h5
import warnings
import glob
import sys
import ast
from pathlib import Path

#if in debug only 10 frames of traj are considered
debug = False

############################################################################################################
######################################           Class           ###########################################
############################################################################################################

class TransitionNetworks:
    topology = None
    trajectory = None
    state = []
    nProt = 0
    ResProt = 0  
    nLig = 0
    ResLig = 0
    multitraj = False
    traj_suf = '.xtc'

    
    
    """The descriptors and corresponding cutoffs to use for constructing the transition matrix."""
    descriptors = {}
    
    def __init__(self, top=topology, traj=trajectory, \
                 nProt=nProt, ResProt=ResProt, nLig=nLig, ResLig=ResLig,\
                 desc=descriptors, state=state, res=1,\
                 traj_suf = traj_suf):

        self.top = top

        """ Check if single traj exists otherwise creat list with all trj that match the prefix"""
        path_TM = Path(traj)

        if path_TM.is_file():
            print('Single trajctory input')
            self.traj_names = [traj]
        
        else:
            """ If input is not single file, check for trajctories with the defined prefix """
            file_names = glob.glob("{}*".format(traj)+traj_suf)  
            N_if=len(file_names)
            print('Multi trajectory input with ',N_if,' trajectories')
            self.traj_names = file_names
        
        """ Initialize MDA """
        self.universe = md.Universe(self.top, self.traj_names[0])

        """ Number of Protein/Ligands and Number of Residues within Protein/Ligand """
        self.nProt = nProt
        self.ResProt = ResProt
        self.nLig = nLig
        self.ResLig = ResLig


        """Generate array to connect atom indices with residue names."""
        self.resnames = self.universe.atoms.resnames
        self.N = len(self.resnames)
        
        """Generate array to connect atom indices with residue indices."""
        self.resindices = self.universe.atoms.resindices

        """ Generrate helper list to split position list of atoms according to residues """
        ind_sum = []
        a = 0
        for i in range(ResProt):
            a += np.count_nonzero(self.resindices == i)
            ind_sum.append(a)

        self.Prot_ind = ind_sum
        
        """ resolution or binwidth for EtE/Rg """
        self.res = res

        """Sort the descriptors dictionary by cutoffs in decreasing order."""
        self.descriptors = {k: v for k, v in sorted(desc.items(), key=lambda item: -item[1])}

        """Generate array with cutoff values in decreasing order."""
        self.cutoffs = np.unique(list(self.descriptors.values()))[::-1]

        self.state = state

        """ Set the full Distance Matrix between Residues and Distance Matrix between chains
            to False as it was not calculated it.
            pp: Protein-Protein
            pl: Protein-Ligand """
        self.Mpp_res = False
        self.Mpp_chain = False
        self.Mpl_res = False
        self.Mpl_chain = False
 
###########################################################################################################
###############################   Generate Transition Matrix   ############################################
###########################################################################################################

    def GenerateTransitionMatrix(self,writetrajectory=False ,statetrjName='state_trj.txt'):
        """Generate the key names for the state defining functions, which may either be composed
        of the descriptors name or of the descriptors name and the associated cutoff value."""
        descriptorKeys = []
        for descriptorName in self.state:
            try:
                descriptorKeys.append(descriptorName + str(self.descriptors[descriptorName]))
            except KeyError:
                descriptorKeys.append(descriptorName)

        """ Initialize state list and dictionary """
        differentStatesList = []
        populationOfStatesDict = {}
        transitionMatrixDict = {}
        countIdx = 0
        states = []

        """ write trajectory of states to be able to identify frames according to states """ 
        if writetrajectory == True:
            f = open(statetrjName,'w')
            f.write('/ states: '+str(self.state))
        
        ''' Cycle through all input trajectories'''
        for traj in self.traj_names:
            print('Processing: '+traj)

            if writetrajectory == True:
                f.write(traj+' \n')

            """ Load traj into MDA """
            self.universe = md.Universe(self.top, traj)
            self.nFrames = self.universe.trajectory.n_frames

            ############################
            ###  Full Traj Analysis  ###
            ############################

            """Check if Secondary Structure needs to be calculated"""
            if any("residuesIn" in s for s in self.state):
                print('Calculating Secondary Structure')
                """Load the trajectory using mdtraj as 'trajec'."""
                trajec = mdtraj.load(traj, top=self.top)
                """Calculated Secoundary structure"""
                self.SecondStruct = mdtraj.compute_dssp(trajec, simplified=True) 

            elif any("OrderParameter" in s for s in self.state):
                print('Calculating Secondary Structure')
                """Load the trajectory using mdtraj as 'trajec'."""
                trajec = mdtraj.load(traj, top=self.top)
                """Calculated Secoundary structure"""
                self.SecondStruct = mdtraj.compute_dssp(trajec, simplified=True)

            """For this function, the package mdtraj is used as it enables to compute the secondary
            structure using dssp. Please note that the simplified dssp codes ('simplified=True') 
            are ‘H’== Helix (either of the ‘H’, or ‘G’, or 'I' Helix codes, which are ‘H’ : Alpha helix
            ‘G’ : 3-helix, 'I' : 5-helix participates in helical ladder. This is used when iterating over all frames""" 

            """ If the descriptor intermolecularSaltBridge is used, before going trough the trajctory,
                the pairs of atoms which might form saltbridges are determined """
            if any("intermolecularSaltBridge" in s for s in self.state):
                print('Determine possible salt bridge forming atom pairs')
                self._FindSaltBridgePartner()


            """ Calc Ramachandran kmeans cluster for whole trajectory  """
            if any("Ramachandran" in s for s in self.state):
                print('Calc Ramachandran kmeans cluster')
                self._Ramachandran()
        
            
            ########################
            ####  Check Frames  ####
            ########################
            """ Iterate over all frames and determine the coresponding state """
            print('Checking States')
            for frame in range(self.nFrames):
                self.universe.trajectory[frame]
                """Extract the states in every frame."""
                stateInFrame = []
            
                """ Set the full Distance Matrix between Residues and Distance Matrix between chains
                    to False """
                self.Mpp_res = None
                self.Mpp_chain = False
                self.Mpl_res = False
                self.Mpl_chain = False

                """loop over all given keys and determine the coresponding state"""
                for key in self.state:

                    ############################################
                    ##### Secondary Structure descriptors ######
                    ############################################

                    if key == 'residuesInHelix':
                        stateInFrame.append(np.count_nonzero(self.SecondStruct[frame] == 'H'))
                    if key == 'residuesInBeta':
                        stateInFrame.append(np.count_nonzero(self.SecondStruct[frame] == 'E'))
                    if key == 'residuesInCoil':
                        stateInFrame.append(np.count_nonzero(self.SecondStruct[frame] == 'C'))

                    ############################################
                    #### Measure of Compactness descriptors ####
                    ############################################

                    """ NC-distance  """

                    if key == 'EndToEnd' : 
                        ete = self._EndToEndDistance()
                        d = self.res * int(ete/self.res)
                        #print('EtE: ', d, ete)    
                        stateInFrame.append(d)         

                    """ Radius of Gyration """ 

                    if key == 'Rg' : 
                        Rg = self._Gyrate()
                        d = self.res * int(Rg/self.res)
                        stateInFrame.append(d)

                    if key == 'CompactnessFactor':

                        d = self._CompactnessFactor()

                        stateInFrame.append(d)

                    ############################################
                    ########### Contact descriptors ############
                    ############################################

                    """ Number of Protein-Ligand Contacts
                        define 'desc = {'ProtLigContacts': cutoff}' in your input file with cutoff beeing a float
                        add state = [..., 'ProtLigContacts'] to your list of states in the input file
                        The number of Contacts are considered between each residue and building block of the Ligand """
                    if key == 'ProtLigContacts':

                        # If not yet calculated calc distance Matrix
                        if type(self.Mpl_res) is not  np.ndarray:
                            self._DistanceProteinLigand()
                        cutoff = float(self.descriptors['ProtLigContacts'])
                        d = self._ContactPairs(self.Mpl_res, cutoff)
                        stateInFrame.append(d)


                    """ Number of Protein-Protein Contacts
                        define 'desc = {'ProtProtContacts': cutoff}' in your input file with cutoff beeing a float
                        add state = [..., 'ProtProtContacts'] to your list of states in the input file
                        The number of Contacts are considered between each residue-residue pair """
                    if key == 'ProtProtContacts':

                        # If not yet calculated calc distance Matrix
                        if type(self.Mpp_res) is not  np.ndarray:
                            self._DistanceProteinProtein()
                        cutoff = float(self.descriptors['ProtProtContacts'])
                        d = self._ContactPairs(self.Mpp_res, cutoff)
                        stateInFrame.append(d)

                    """ Number of Protein-Protein chain contacts
                        define 'desc = {'ProtProtContactsChain': cutoff}' in your input file with cutoff beeing a float
                        add state = [..., 'ProtProtContactsChain'] to your list of states in the input file
                        The number of Contacts are considered between pairs of protein chains """
                    if key == 'ProtProtContactsChain':

                        # If not yet calculated calc distance Matrix
                        if type(self.Mpp_chain) is not  np.ndarray:
                            self._DistanceProteinProtein()
                        cutoff = float(self.descriptors['ProtProtContactsChain'])
                        d = self._ContactPairs(self.Mpp_chain, cutoff)
                        stateInFrame.append(d)

                    """ Calculates number of hydrophobic contacts between pairs of proteins """
                    if key == 'hydrophobicContacts':
                        # If not yet calculated calc distance Matrix
                        if type(self.Mpp_res) is not  np.ndarray:
                            self._DistanceProteinProtein()

                        d = self._residuesWithAttribute(descriptorName='hydrophobicContacts', attribute='hydrophobic')

                        stateInFrame.append(d)

                    """ Calculates number of hydrophobic contacts between pairs of proteins """
                    if key == 'polarContacts':
                        # If not yet calculated calc distance Matrix
                        if type(self.Mpp_res) is not  np.ndarray:
                            self._DistanceProteinProtein()

                        d = self._residuesWithAttribute(descriptorName='polarContacts', attribute='polar')

                        stateInFrame.append(d)
                    ############################################
                    ############  Oligomeric Size  #############
                    ############################################

                    """ Calculates the size of the largest Oligomer from the chain distance matrix (self.Mpp_chain)
                        Use the descriptor  'desc = {'OligomericSize': cutoff}' and 'state = [... , 'OligomericSize']
                        in your input. """
                    if key == 'OligomericSize':
                       # If not yet calculated calc distance Matrix
                        if type(self.Mpp_chain) is not np.ndarray:
                            self._DistanceProteinProtein()

                        """ Get the cutoff value for the descriptor function oligomeric size """
                        cutoff = self.descriptors['OligomericSize']
                        d = self._OligomericSize(cutoff)
                        stateInFrame.append(d)

                    ############################################
                    ############    Saltbridges    #############
                    ############################################

                    """ Calculste number of intermolecular salt bridges """
                    if key == 'intermolecularSaltBridge':
                        cutoff = self.descriptors['intermolecularSaltBridge']
                        d = self._CalcNumberSaltBridges(cutoff)

                        stateInFrame.append(d)

                    ############################################
                    ############ Order descriptors #############
                    ############################################

                    """ Calculate Order Parameter. P1: polar order parameter, P2: nematic order parameter"""
                    if key == 'OrderParameter':

                        d = self._OrderParameter()

                        P1 = round(d[0],1)
                        P2 = round(d[1],1)

                        stateInFrame.append(P1)
                        stateInFrame.append(P2)

                    ############################################
                    ###########    Ramachandran    #############
                    ############################################

                    """ Ramachandran kmeans value """

                    if key == 'Ramachandran' : 
                        feature = self.rama_list[frame]  
                        stateInFrame.append(feature)


                    ############################################
                    ########### Custom descriptors #############
                    ############################################

                    """ If you want to add a decriptor specify this here.
                        You can then add [..., custom] to your list of states in the input file """

                    if key == 'custom' : 
                        feature = _CustomDescriptor()  
                        stateInFrame.append(feature)


                """ If a state did not occur yet, append it to the list 'differenStatesList',
                   update its population and generate an entry in the dictionary 'transitionMatrixDict'
                   to later relate the entries in the 'transitionMatrix' with transitions between states,
                   in other words to relate indices <-> states."""

                if (stateInFrame not in differentStatesList):
                    differentStatesList.append(stateInFrame)
                    populationOfStatesDict.update({tuple(stateInFrame): 1})
                    transitionMatrixDict.update({tuple(stateInFrame): countIdx})
                    countIdx += 1

                    """If a state is already known, increase its population by one
                    update its population."""

                else:
                    population = populationOfStatesDict[tuple(stateInFrame)]
                    population += 1
                    populationOfStatesDict.update({tuple(stateInFrame): population})


                if frame%int(self.nFrames/10)==0:
                    print(int(frame/self.nFrames*100),'% of frames processed')
            
                """Append the states observed in this frame to the overall array 'states'."""
                states.append(tuple(stateInFrame))
                      
                if writetrajectory == True:
                    f.write(str(frame)+'\t'+str(stateInFrame)+'\t\n')

                if debug == True:
                    if frame == 10: 
                        break      

        if writetrajectory == True:
            f.close()


        """Get the number of different states which have been observed along the
        trajectory."""
        differentStates = len(differentStatesList)
        states = np.array(states)
        
        """Initialize the transition matrix."""
        transitionMatrix = np.zeros((differentStates, differentStates), dtype=int)
        
        """Fill the transition matrix with transition values by counting all observed
        transitions between two states."""
        
        stateHistory = states
        for state1, state2 in zip(stateHistory[:-1], stateHistory[1:]):
                idx1 = transitionMatrixDict[tuple(state1)]
                idx2 = transitionMatrixDict[tuple(state2)]
                transitionMatrix[idx1][idx2] += 1

        return (transitionMatrix, transitionMatrixDict)

####################################################################################################### 
#######################################       Generate Network      ###################################
#######################################################################################################
    
    def GenerateNetwork(self, minPopulation=0.0, minTransition=0.0, gexfName="network.gexf", TransitionMatrixName='my_TransitionMatrix.npy', DictionaryName='my_TransitionMatrixDict.txt', statetrjName='state_trj.txt'):
        """This function generates a .gexf file which can be visualized using the
        program 'Gephi'. This files contains the following information:
            - population of a state can be visualized by a node's size
            - the name of a node is the state
            - the amount of transition between two states is encoded in the
              line thickness
            - the direction of the transition is read to be clockwise.
            
        Futhermore the user is asked to give 'minPopulation' as input, which is a threshold
        for only considering nodes possesing with at least 'minPopulation'*100 percent population
        of the maximum observed population"""
        
        """ Check if Transition Matrix already exists and read it in"""
        path_TM = Path(TransitionMatrixName)

        if path_TM.is_file():
            print('Loading existing Transition Matrix')
            transitionMatrix = np.load(TransitionMatrixName)
            transitionMatrixDict = self.LoadDict(DictionaryName)

        else:            
            print('Calculating Transition Matrix')
            transitionMatrix, transitionMatrixDict = self.GenerateTransitionMatrix(writetrajectory=True, statetrjName=statetrjName)

            """ Save Matrix and dictionary """
            np.save(TransitionMatrixName,transitionMatrix)
            self.SaveDict(DictionaryName,transitionMatrixDict)


        print('Building Network')
        transitionMatrixNonDiagonal = transitionMatrix.copy()
        for idx,_ in enumerate(transitionMatrixNonDiagonal):
            transitionMatrixNonDiagonal[idx][idx] = 0
                
        
        """Get the maximum values for a node pouplation and a transition."""
        maxPopulation = max(np.diag(transitionMatrix))
        maxTransition = np.max(transitionMatrixNonDiagonal)
        
        """Generate a dictionary with nodes and normalized population, which are at least greater than
        'minPopulation'."""
        nodesDict = {}
        for state, size in zip(transitionMatrixDict.keys(), np.diagonal(transitionMatrix)):
            fraction = size/maxPopulation
            if (fraction >= minPopulation):
                nodesDict.update({state: size})

        """Only consider normalized transitions with a value of at least 'minTransition'."""
        edgesDict = {}
        for state1, (idx1, row) in zip(transitionMatrixDict.keys(), enumerate(transitionMatrix)):
            for state2, (idx2, transition) in zip(transitionMatrixDict.keys(), enumerate(row)):
                if (idx1 != idx2 and transition != 0):
                    if (state1 in nodesDict.keys() and state2 in nodesDict.keys()):
                        fraction = transition/maxTransition
                        if (fraction >= minTransition):
                            edgesDict.update({(state1, state2): transition})
                            

        G = nx.DiGraph()
        for k, v in nodesDict.items():
            G.add_node(k, size=float(v))
        for k, v in edgesDict.items():
            G.add_edge(k[0], k[1], weight=float(v))
        nx.draw(G)
        nx.write_gexf(G, gexfName)
        #plt.show()

####################################################################################################### 
#######################################     Descriptor Functions    ###################################
#######################################################################################################

    """ Just specify your descriptor function here and add it to the loop in GenerateTransitionMatrix()
        as described above """
    def _CustomDescriptor(self):

        print('This could be your function')   

        return 0

########################
###### Contacts ########
########################

    """ Calculate all residue-residue distances between all Proteins and saves the matrix of square distances
        to the class """
    def _DistanceProteinProtein(self):

        frame = self.universe.trajectory
        box = self.universe.dimensions

        self.Mpp_res = np.zeros((self.nProt*self.ResProt,self.nProt*self.ResProt))
        self.Mpp_chain = np.zeros((self.nProt,self.nProt)) 

        for i in range(self.nProt):
            for j in range(i+1,self.nProt):

                """ Read out the Atom postions of one pair of Proteins and split the position vector
                    according to their respective residues """
                posA = self.universe.residues[i*self.ResProt:(i+1)*self.ResProt].atoms.positions                   
                posA = np.split(posA,self.Prot_ind[:-1])
                posB = self.universe.residues[j*self.ResProt:(j+1)*self.ResProt].atoms.positions 
                posB = np.split(posB,self.Prot_ind[:-1])

                """ Calc square Distance Matrix """
                M_sq = self._DistMatrix(posA,posB,box)

                """ save block matrix """
                self.Mpp_res[i*self.ResProt:(i+1)*self.ResProt,j*self.ResProt:(j+1)*self.ResProt] = M_sq

                
                self.Mpp_chain[i,j] = np.amin(M_sq)

    """ Calculate all residue-ligand distances between Proteins and Ligands saves the matrix of square distances
        to the class.
        Consideres each ligand as one object thus each residue of the protein can only have up to one contact 
        with the Ligand """
    def _DistanceProteinLigand(self):

        frame = self.universe.trajectory
        box = self.universe.dimensions

        self.Mpl_res = np.zeros((self.nProt*self.ResProt,self.nLig))
        self.Mpl_chain = np.zeros((self.nProt,self.nLig)) 

        for i in range(self.nProt):
            for j in range(self.nLig):

                """ Read out the Atom postions of one pair of Proteins and split the position vector
                    according to their respective residues """
                posA = self.universe.residues[i*self.ResProt:(i+1)*self.ResProt].atoms.positions                   
                posA = np.split(posA,self.Prot_ind[:-1])
                posB = self.universe.residues[(self.nProt*self.ResProt + j*self.ResLig):(self.nProt*self.ResProt+ (j+1)*self.ResLig)].atoms.positions 
                posB = [posB]
                """ Calc square Distance Matrix """
                M_sq = self._DistMatrix(posA,posB,box)


                """ save block matrix """
                self.Mpl_res[i*self.ResProt:(i+1)*self.ResProt,j*self.ResLig:(j+1)*self.ResLig] = M_sq

                
                self.Mpl_chain[i,j] = np.amin(M_sq)

      

    """ Calculates the number of Contacts for a given ContactMatrix"""
    def _ContactPairs(self,M_c, cutoff):

        """ Check for zero values, which are entries that were not calculated and should be disregarded
            In the Distance calculation, usually only the lower triangle matrix is calculated """
        mask0 = M_c != 0

        """ Check how many Protein-Protein squared distances are below squared cutoff"""
        mask_c = M_c < cutoff*cutoff

        """ Check overlap between non-zero entries that are below the cutoff value """
        mask = mask_c == mask0
 
        Contacts = np.sum(mask.astype(int))


        return Contacts

    """ Calculates how many contacts between residues with a certain attribute are present """
    def _residuesWithAttribute(self, descriptorName=None, attribute=None):
        """The user can define an attribute group by augmentation or modification of
        the 'attributes' dictionary."""
        attributes = {'hydrophobic': ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'MET', 'TRP'],
                      'polar': ['SER', 'THR', 'CYS', 'ASN', 'GLN', 'TYR']}
        residuesWithAttribute = attributes[attribute]    
    
        """Get the cutoff value for the descriptor function."""
        cutoff = self.descriptors[descriptorName]

        """ Go through residue names of involved proteins and create array where 1 represents that the residue
            belongs to the group of residues with the specified attribute and 0 it does not """
        residues = self.universe.residues.resnames
        res_attribute = []
        for res in residues:
            match = False
            for att in residuesWithAttribute:
                if res == att:
                    res_attribute.append(1)
                    match = True

            if match == False:
                res_attribute.append(0)

        res_attribute = np.array(res_attribute)

        """ Create Matrix with all possible contacts between atoms of the group represented by a 1
            by building the outer product of res_attribute """
        grid = np.outer(res_attribute,res_attribute)
        np.set_printoptions(threshold=sys.maxsize)

        conv_dist = self.Mpp_res*grid

        d = self._ContactPairs(conv_dist,cutoff)
        return d


########################
##### Saltbridge  ######
########################

    """ Determine all pairs of atoms which might form intermolecular Salt Bridges """                
    def _FindSaltBridgePartner(self):
        """At first an atom group containing all atoms of chain A is generated. It is important to
        mention that all chains have to be identical!"""
        atomsChainA = self.universe.residues[:self.ResProt].atoms
        self.nAtomsPerChain = len(atomsChainA.positions)
        """Now the bonds between the selected atoms are guessed and the elements of these atoms are
        identified."""
        
        "***!!Could be optimized using mdtraj bonds!!***'"
        atomsChainA.guess_bonds()
        elements = [atom.type for atom in atomsChainA]

        """Initialize positive 'pos' and negative 'neg' charge of that chain and the lists where
        the charged atoms are stored, positively charged atom indices are stored in 
        'saltBridgeAcceptorsChainA', negatively charged atom indices are stored in
        saltBridgeDonorsChainA, respectively."""
        pos = 0
        saltBridgeAcceptorsChainA = list()
        neg = 0
        saltBridgeDonorsChainA = list()
        
        for atomIdx, element in enumerate(elements):
            
            """If atom with index 'atomIdx' is a nitrogen atom and has four binding partners 
            (corresponding to a charge of +1) raise the counter of positively charged atoms 
            'pos' by one and add 'atomIdx' to the list of positively charged atoms 
            'saltBridgeAcceptorsChainA'.""" 
            if (element == 'N'):
                bondingDegree = 0
                for bond in atomsChainA.bonds:
                    if (atomIdx in bond.indices):
                        bondingDegree += 1
                
                if (bondingDegree == 4):
                    saltBridgeAcceptorsChainA.append(atomIdx)
                    pos += 1

            elif (element == 'O'):
                """If atom with index 'atomIdx' is an oxygen atom and has only one binding partners
                with a single bond (corresponding to a charge of -1) and the next atom is also an
                oxygen atom (allows identification of carboxylic groups COO-) lower the counter of 
                negatively charged atoms 'neg' by one and add  'atomIdx' to the list of negatively
                charged atoms 'saltBridgeDonorsChainA'."""   
                bondingDegree = 0
                for bond in atomsChainA.bonds:
                    if (atomIdx in bond.indices):
                        try:
                            if (elements[atomIdx + 1] == 'O'):
                                bondingDegree += 1
                                
                        except IndexError:
                            pass
                        
                """Also append the index of the next atom, because the charge can be distributed among
                both oxygen atoms."""
                if (bondingDegree == 1):
                    saltBridgeDonorsChainA.append(atomIdx)
                    neg -= 1
        
        """Total charge of chain A is simply the sum of positively and negatively counted atoms."""
        chargeChainA = pos + neg
        
        saltBridgeAcceptorsChainA = np.array(saltBridgeAcceptorsChainA)
        saltBridgeDonorsChainA = np.array(saltBridgeDonorsChainA)
        
        """Generate a list with atom indices of possible salt bridge contact pairs."""
        pairs = []

        for i in range(self.nProt):
            """Exclude intramolecular salt bridge pairs."""
            for j in range(i+1,self.nProt):

                for acc in saltBridgeAcceptorsChainA:
                    for don in saltBridgeDonorsChainA:
                        """ Add both interaction between (ch1-ch2 and ch2-ch1)
                            one might want to consider both O of COO- group as donors """
                        pairs.append([acc+i*self.nAtomsPerChain,don+j*self.nAtomsPerChain])

                        pairs.append([acc+j*self.nAtomsPerChain,don+i*self.nAtomsPerChain])


        self.SaltPairs = pairs

    """ Calculate the inter moecular Salt Bridges from the list of Salt Pairs
        In order to work  self._FindSaltBridgePartner() has to be executed once"""
    def _CalcNumberSaltBridges(self,cutoff):

        box = self.universe.dimensions
        Contacts = 0 

        for pair in self.SaltPairs:
            pos1 = self.universe.atoms[pair[0]].position
            pos2 = self.universe.atoms[pair[1]].position

            dsq = self._SquaredDistance(pos1,pos2,box)

            if dsq < cutoff*cutoff:
                Contacts += 1


        return Contacts

########################
### Order Parameter ####
########################

    """ Calculate the polar/nematic order parameter P1/P2 for the orientation of the molecules.
        See: 'Thermodynamic analysis of structural transitions during GNNQQNY aggregation'
        from Kenneth L. Osborne,  Michael Bachmann, and Birgit Strodel published in
        Proteins 2013; 81:1141–1155. """
    def _OrderParameter(self):

        NCvec = np.zeros((self.nProt,3))
        
        Npos = self.universe.select_atoms("name N").positions
        Cpos = self.universe.select_atoms("name C").positions

        for i in range(self.nProt):
            
            Nvec = Npos[i*self.ResProt]
            Cvec = Cpos[(i+1)*self.ResProt-1] # -1 as list starts from 0
            diff = Cvec - Nvec

            NCvec[i] = diff/np.linalg.norm(diff)

        """Compute the director using the normalized vectors."""
        director = self._director(NCvec) 

        """Compute the polar and nematic order parameters."""
        polarOrderParameter = 0
        nematicOrderParameter = 0
        for vector in NCvec:
            polarOrderParameter += np.matmul(vector, director)
            nematicOrderParameter += 3*np.matmul(vector, director)**2 - 1
            
        """Use the absolut value, because negative only occur as result of the algorithm's
        used for computing the eigenvectors."""
        polarOrderParameter = abs(polarOrderParameter/len(NCvec))
        nematicOrderParameter = abs(nematicOrderParameter/(2*len(NCvec)))
        return (polarOrderParameter, nematicOrderParameter)       

        
    def _director(self, vectors):
        """Computation of the direction 'director' most molecules are aligned to. The director
        is equal to the eigenvector corresponding to the largest eigenvalue of the ordering
        matrix Q."""
        Qsum = np.zeros((3, 3))
        
        """Compute Q for every vector."""
        for v in vectors:
            vector = v
            
            Q = np.zeros((3, 3))
            for a in range(3):
              
                """Since Q is a symmetric matrix, only calculate lower triangle"""
                for b in range(a,3):
                    if (a == b):
                        Qab = 3*vector[a]*vector[b] - 1
                        Q[a][b] = Qab
                        Q[b][a] = Qab
                    else:
                        Qab = 3*vector[a]*vector[b]
                        Q[a][b] = Qab
                        Q[b][a] = Qab 

            Qsum += Q      
        
        """Compute the averaged ordering matrix Q."""
        Qsum /= (2*len(vectors))

        """Determine the eigenvalues and associated eigenvectors of Q."""
        eigenValues, eigenVectors = np.linalg.eigh(Qsum)
        
        """Sort the eigenvalues and associated eigenvectors in decreasing order."""
        idx = eigenValues.argsort()[::-1]   
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]

        """Return the director by extracting the eigenvector correspoding to the largest
        eigenvalue."""
        director = eigenVectors[:,0]   
        return (director)

 
########################
###### Oligomer ########
########################
       
    """ Calculate the size of the largest oligomer within the system """
    def _OligomericSize(self,cutoff):

        """ Determine Contact Matrix between pairs of chains (Same Routine as self._ContactPairs)""" 
        mask0 = self.Mpp_chain != 0
        mask_c = self.Mpp_chain < cutoff*cutoff
        mask = mask_c == mask0
 
        #mask = mask.astype(int)
        ''' matrix with index in lines '''
        mask_index = np.ones(mask.shape)
        mask_index[:] = np.arange(self.nProt)
     
        oligomer = [set([i]) for i in range(self.nProt)] 
        
        for i in range(self.nProt):
            contacts = mask_index[i][mask[i]]
            for indx in contacts:
                oligomer[i].add(int(indx))
                 

        
        for i in range(self.nProt):
            for j in oligomer[i]:
                if i != j:
                    united = oligomer[i].union(oligomer[j])
                    oligomer[i] = united
                    oligomer[j] = united

        #size = [len(oligomer[i]) for i in range(self.nProt)]

        oligomer = [tuple(i) for i in oligomer]
        oligomer_set = np.array(list(set(oligomer)))
        size = np.array([len(i) for i in oligomer_set])

        idx = np.argsort(size)[::-1]

        oligomer_set = oligomer_set[idx]
        oligomer_size = size[idx]

        return oligomer_size[0]
        
########################
##### Compactness ######
########################

    ''' Calculates the distance between the N and C terminus of each protein and returns the average value'''         
    def _EndToEndDistance(self):
        frame = self.universe.trajectory
        box = self.universe.dimensions

        Npos = self.universe.select_atoms("name N").positions
        Cpos = self.universe.select_atoms("name C").positions

        EtE = 0

        for i in range(self.nProt):

            Ni = Npos[i*self.ResProt]
            Ci = Cpos[(i+1)*self.ResProt-1]

            """ calc square distance between N and C """
            d = self._SquaredDistance(Ni,Ci,box)

            EtE += d

        return EtE/self.nProt

    ''' Calculates average Molecules Radius of Gyration'''         
    def _Gyrate(self):
        frame = self.universe.trajectory
        box = self.universe.dimensions

        Rg = 0
        for i in range(self.nProt):
            """ calculate mean vector of positions """
            res = self.universe.residues[i*self.ResProt:(i+1)*self.ResProt]
            rm = np.mean(res.atoms.positions,axis=0)

            rmsd = res.atoms.positions-rm
            rsq = 0
            """ loop over all residues of the Protein and calculate (r_i-rm)**2 """
            for v in rmsd:
                rsq += np.dot(v,v)

            Rg += np.sqrt(rsq/len(rmsd))
        
        return Rg/self.nProt

    ''' Calculate the compactness of the system based on the moment of inertia '''
    def _CompactnessFactor(self):
        atoms = self.universe.atoms

        """Calculate the moment of inertia tensor for the structure and its
        eigenvalues using periodic boundary conditions."""
        momentOfInertiaTensor = atoms.moment_of_inertia(pbc=True)
        eigenvalues = np.linalg.eigvals(momentOfInertiaTensor)

        """Calculate the 'compactness' based on the lowest and highest eigenvalue."""
        compactness = abs(round(10*min(eigenvalues)/max(eigenvalues)))

        print(compactness)
        return compactness      

########################
##### Ramachandran #####
########################

    """This function is only valid for monomers! Keep in mind that cluster indices can differ."""
    def _Ramachandran(self):
        warnings.filterwarnings("ignore", message="Cannot determine phi and psi angles for the first or last residues")
        
        allAtoms = self.universe.select_atoms("protein")

        phis, psis = Ramachandran(allAtoms).run().angles.T 
        phis = phis.T
        psis = psis.T
        
        X = list()
        for xArr, yArr in zip(phis[:, :], psis[:, :]):
            for x, y in zip(xArr, yArr):
                X.append([x, y])
        X = np.array(X)
        
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        xCluster, yCluster = kmeans.cluster_centers_.T
        
        kmean_list = np.zeros(len(phis))
        for frameIdx, (xArr, yArr) in enumerate(zip(phis[:, :], psis[:, :])):
            value = 0
            for x, y in zip(xArr, yArr):
                value += kmeans.predict([[x, y]])[0] 

            kmean_list[frameIdx] = value


        self.rama_list = kmean_list
####################################################################################################### 
#######################################       Utility Functions     ###################################
#######################################################################################################
   
    ''' Calculate the total number of self transitions ie. total 'size' '''
    def SumDiagonal(self,Matrix=False,TransitionMatrixName='my_TransitionMatrix.npy'):

        ''' Load Matrix '''
        if Matrix == True:
            TM = Matrix
        else:
            """ Check if Transition Matrix already exists and read it in"""
            path_TM = Path(TransitionMatrixName)
            if path_TM.is_file():
                print('Loading existing Transition Matrix')
                TM = np.load(TransitionMatrixName)

            else:            
                print('Please provide either a .npy file contaonong a TN matrix or directly to the function')
        
        diag = np.diagonal(TM)
        totsize = np.sum(diag)

        return totsize

   
    def PrintFrame(self):
        print(self.universe.trajectory.time)
 
    ''' save load dictionary '''
    def SaveDict(self,outname,dictionary):

        with open(outname, 'w') as f:
            
            f.write('@ Transition Matrix Dictionary \n')
            f.write('@ state population \n')
            for state, indx in dictionary.items():
                f.write(str(state)+'\t'+str(indx)+'\t\n')

    def LoadDict(self,inname):
        dictionary = {}
        with open(inname, 'r') as f:
            for line in f:
                if not line.startswith('@'):
                    l = line.split('\t')

                    dictionary.update({eval(l[0]): int(l[1])})

        return dictionary


########################
#### Correltaions ######
########################

    def CorrelationCoefficients(self,trjpath = 'state_trj.txt'):

        states = []
        with open(trjpath,'r') as f:

            for line in f:

                if not line.startswith('/'):

                    x = line.split('\t')
                    s = ast.literal_eval(x[1])
                    states.append(np.array(s))

        Ndesc = len(states[0])
        Nstates = len(states)
        states = np.array(states)

        for i in range(Ndesc):
            for j in range(i+1,Ndesc):

                d1 = states[:,i]
                d2 = states[:,j]

                corr = np.corrcoef(d1, d2)
                cross_corr = corr[0][1]

                print('Correlation {} {}: {}'.format(self.state[i],self.state[j],cross_corr))

        
                
########################
###### Compiled ########
########################

    @staticmethod                        
    @jit(nopython=True, parallel=True)
    def _DistMatrix(pos1,pos2,box):
        """Calculates ContactMatrix between atoms of two residues
        
        Parameters
        ----------
        pos1: ndarray
            Nx3 array with 3D coordinates of atoms of residue 1
        
        pos2: ndarray
            Nx3 array with 3D coordinates of atoms of residue 2

        box: ndarray
            3x1 array with box coordinate vectors
            
        Returns
        -------
        M: ndarray
            matrix of size(len(pos1),len(pos2))
        """
        
        nres1 = len(pos1)
        nres2 = len(pos2)

        # create empty contact matrix
        M_sq = np.zeros((nres1,nres2))

        # loop over all residues group A and group B to calculate their distance and check
        # weather or not they are in contact 
        # (only iterate over upper triangular matrix as it is symmetric)
        for i in prange(nres1):

            for j in range(nres2):

                resA = pos1[i]
                nA = len(resA)
                resB = pos2[j]
                nB = len(resB)
                
                # create empty contact matrix
                M_d = np.zeros((nA,nB))
                
                for k in range(nA):
                    for l in range(nB):
                        
                        dx = resA[k,0] - resB[l,0]
                        dy = resA[k,1] - resB[l,1]
                        dz = resA[k,2] - resB[l,2]
                
                        # pbc
                        dx -= box[0] * round(dx/box[0])
                        dy -= box[1] * round(dy/box[1])
                        dz -= box[2] * round(dz/box[2])
                
                        dsq = dx*dx + dy*dy + dz*dz
                    
                        M_d[k,l] = dsq

                d_min = np.amin(M_d)

                #M[i,j] = dmin**0.5
                M_sq[i,j] = d_min

        return M_sq

    @staticmethod                        
    @jit(nopython=True)
    def _SquaredDistance(pos1,pos2,box):
        """Calculates ContactMatrix between atoms of two residues
        
        Parameters
        ----------
        pos1: ndarray
            3x1 array with 3D coordinates of an atom
        
        pos2: ndarray
            3x1 array with 3D coordinates of an atom

        box: ndarray
            3x1 array with box coordinate vectors

            
        Returns
        -------
        d: scalar
            square distance
        """
        
                        
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        dz = pos1[2] - pos2[2]

        # pbc
        dx -= box[0] * round(dx/box[0])
        dy -= box[1] * round(dy/box[1])
        dz -= box[2] * round(dz/box[2])

        dsq = dx*dx + dy*dy + dz*dz

        return dsq







