# -*- coding: utf-8 -*-
"""
ATRANET (Automated TRAnsition NETworks) code for caclulating TNs from MD data.
Originaly written by:
Improved and modified by: Moritz Schäffler

The script is designed in a way that it allows the user easily to add further
descriptor functions to the script if they seem to be more beneficial for the system under
investigation. To perform the TNs analysis, topology and trajectory files of
all-atom MD simulations with any number of peptide chains have to be provided.
From this, the script will produce a file that contains the transition
matrix and the state’s populations (”size”), which can be visualized
by network visualization software like Gephi.

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
import mdtraj # necessary for dssp
import networkx as nx
import numpy as np
import h5py as h5
import warnings
import glob
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
    nLig = 0
    multitraj = False

    
    
    """The descriptors and corresponding cutoffs to use for constructing the transition matrix."""
    descriptors = {}
    
    def __init__(self, top=topology, traj=trajectory, nProt=nProt,\
                 nLig=nLig,desc=descriptors, state=state, res=1,\
                 h5File='transitionNetwork.hdf5'):

        self.top = top

        """ Check if single traj exists otherwise creat list with all trj that match the prefix"""
        path_TM = Path(traj)

        if path_TM.is_file():
            print('Single trajctory input')
            self.traj_names = [traj]
        
        else:
            """ If input is not single file, check for trajctories with the defined prefix """
            file_names = glob.glob("{}*.xtc".format(traj))  
            N_if=len(file_names)
            print('Multi trajectory input with ',N_if,' trajectories')
            self.traj_names = file_names

        """ Number of Residues within Protein/Ligand """
        self.nProt = nProt
        self.nLig = nLig

        self.state = state
        self.h5File = h5File

        """ resolution or binwidth for EtE/Rg """
        self.res = res

        """Sort the descriptors dictionary by cutoffs in decreasing order."""
        self.descriptors = {k: v for k, v in sorted(desc.items(), key=lambda item: -item[1])}

        """Generate array with cutoff values in decreasing order."""
        self.cutoffs = np.unique(list(self.descriptors.values()))[::-1]
 
###########################################################################################################
###############################   Generate Transition Matrix   ############################################
###########################################################################################################

    def GenerateTransitionMatrix(self,writetrajectory=False):
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
            f = open('state_trj.txt','w')
        
        ''' Cycle through all input trajectories'''
        for traj in self.traj_names:
            print('Processing: '+traj)

            if writetrajectory == True:
                f.write(traj+' \n')

            """ Load traj into MDA """
            self.universe = md.Universe(self.top, traj)
            self.nFrames = self.universe.trajectory.n_frames

            """Check if Secondary Structure needs to be calculated"""
            if any("residuesIn" in s for s in self.state):
                print('Calculating Secondary Structure')
                """Load the trajectory using mdtraj as 'trajec'."""
                trajec = mdtraj.load(traj, top=self.top)
                """Calculated Secoundary structure"""
                SecondStruct = mdtraj.compute_dssp(trajec, simplified=True) 

            """For this function, the package mdtraj is used as it enables to compute the secondary
            structure using dssp. Please note that the simplified dssp codes ('simplified=True') 
            are ‘H’== Helix (either of the ‘H’, or ‘G’, or 'I' Helix codes, which are ‘H’ : Alpha helix
            ‘G’ : 3-helix, 'I' : 5-helix participates in helical ladder. This is used when iterating over all frames""" 

        
            """ Iterate over all frames and determine the coresponding state """
            print('Checking States')
            for frame in range(self.nFrames):
                self.universe.trajectory[frame]
                """Extract the states in every frame."""
                stateInFrame = []
            
                """loop over all given keys and determine the coresponding state"""
                for key in self.state:

                    """ Number of Inter Molecular Contacts
                        define 'desc = {'allContacts': cutoff}' in your input file with cutoff beeing a float
                        add state = [..., 'allContacts'] to your list of states in the input file"""

                    if key == 'allContacts':           
                        imCP = self.intermolecularContactPairs(float(self.descriptors['allContacts']))
                    
                        stateInFrame.append(imCP)

                    """ Secondary Structure descriptors """

                    if key == 'residuesInHelix':
                        stateInFrame.append(np.count_nonzero(SecondStruct[frame] == 'H'))
                    if key == 'residuesInBeta':
                        stateInFrame.append(np.count_nonzero(SecondStruct[frame] == 'E'))
                    if key == 'residuesInCoil':
                        stateInFrame.append(np.count_nonzero(SecondStruct[frame] == 'C'))

                    """ NC-distance  """

                    if key == 'EndToEnd' : 
                        ete = self.EndToEndDistance()
                        d = self.res * int(ete/self.res)
                        #print('EtE: ', d, ete)    
                        stateInFrame.append(d)         

                    """ Radius of Gyration """ 

                    if key == 'Rg' : 
                        Rg = self.Gyrate()
                        d = self.res * int(Rg/self.res)
                        stateInFrame.append(d)

                    """ If you want to add a decriptor specify this here.
                        You can then add [..., custom] to your list of states in the input file """

                    if key == 'custom' : 
                        feature = CustomDescriptor()  
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
#######################################     Descriptor Functions    ###################################
#######################################################################################################

    """ Just specify your descriptor function here and add it to the loop in GenerateTransitionMatrix()
        as described above """
    def CustomDescriptor(self):

        print('This could be your function')   

        return 0

        
    '''Calculates the number of Contacts between residues of Chain A and Chain B'''
    def intermolecularContactPairs(self,cutoff):

        frame = self.universe.trajectory
        box = self.universe.dimensions
        
        Contacts = 0
        """ loop over all residues of the Protein and the Ligand to calculate their distance and check
            whether or not they are in contact, depending on the chosen cutoff """
        for i in range(0,self.nProt-1):

            for j in range(self.nProt,self.nProt+self.nLig-1):

                resA = self.universe.residues[i]
                resB = self.universe.residues[j]
                
                """ calc distance matrix of all atoms """
                M_d = distance_array(resA.atoms.positions,resB.atoms.positions, box=box)

                """ Check minimum distance between residues and compare with cutoff"""
                d_min = np.amin(M_d)
                
                if d_min <= cutoff:               
                    Contacts += 1

        return Contacts
        
    ''' Calculates the distance between the first and last Protein residue'''         
    def EndToEndDistance(self):
        frame = self.universe.trajectory
        box = self.universe.dimensions

        resA = self.universe.residues[0]
        resB = self.universe.residues[self.nProt-1]

        """ calc distance matrix of all atoms and extract minimum dist """
        M_d = distance_array(resA.atoms.positions,resB.atoms.positions, box=box)

        d_min = np.amin(M_d)

        return d_min

    ''' Calculates Molecules Radius of Gyration'''         
    def Gyrate(self):
        frame = self.universe.trajectory
        box = self.universe.dimensions

        """ calculate mean vector of positions """
        res = self.universe.residues[0:self.nProt-1]
        rm = np.mean(res.atoms.positions,axis=0)

        rmsd = res.atoms.positions-rm
        rsq = 0
        """ loop over all residues of the Protein and calculate (r_i-rm)**2 """
        for v in rmsd:
            rsq += np.dot(v,v)

        Rg = np.sqrt(rsq/len(rmsd))
        
        return Rg

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

####################################################################################################### 
#######################################       Generate Network      ###################################
#######################################################################################################
    
    def generateNetwork(self, minPopulation=0.0, minTransition=0.0, gexfName="network.gexf", TransitionMatrixName='my_TransitionMatrix.npy', DictionaryName='my_TransitionMatrixDict.txt'):
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
            transitionMatrixDict = self.loadDict(DictionaryName)

        else:            
            print('Calculating Transition Matrix')
            transitionMatrix, transitionMatrixDict = self.GenerateTransitionMatrix(writetrajectory=True)

            """ Save Matrix and dictionary """
            np.save(TransitionMatrixName,transitionMatrix)
            self.saveDict(DictionaryName,transitionMatrixDict)


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
#######################################       Utility Functions     ###################################
#######################################################################################################
    
    ''' save load dictionary '''
    def saveDict(self,outname,dictionary):

        with open(outname, 'w') as f:
            
            f.write('@ Transition Matrix Dictionary \n')
            f.write('@ state population \n')
            for state, indx in dictionary.items():
                f.write(str(state)+'\t'+str(indx)+'\t\n')

    def loadDict(self,inname):
        dictionary = {}
        with open(inname, 'r') as f:
            for line in f:
                if not line.startswith('@'):
                    l = line.split('\t')

                    dictionary.update({eval(l[0]): int(l[1])})

        return dictionary
                







