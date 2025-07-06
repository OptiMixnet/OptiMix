# -*- coding: utf-8 -*-
"""
This .py file includes strategies that a mixnode adversary might consider when corrupting mindnodes 
in mixnets with stratified topologies.
"""
from pulp import LpMinimize, LpProblem, lpSum, LpVariable, PULP_CBC_CMD
import numpy as np
import math

from Routings import Routing

def find_closest_points(data, m):
    if m > len(data):
        raise ValueError("m should be less than or equal to the length of the data list")

    # Divide data into regions
    regions = divide_data_by_region(data)

    # Find the region with the highest concentration
    max_region = max(regions, key=lambda key: len(regions[key]))

    # Randomly select m nodes from the region with the highest concentration
    selected_nodes = random.sample(regions[max_region], m)

    # Get indices of selected nodes in the original data list
    indices_of_selected_nodes = [data.index(node) for node in selected_nodes]

    return selected_nodes, indices_of_selected_nodes




def findindices(prob_dist, s):
    n = len(prob_dist)
    
    # Check if s is greater than the length of the probability distribution
    if s > n:
        raise ValueError("s cannot be greater than the length of the probability distribution")

    # Generate all combinations of indices for subsets of size s
    index_combinations = list(itertools.combinations(range(n), s))

    # Calculate the sum of probabilities for each combination
    subset_sums = [sum(prob_dist[i] for i in indices) for indices in index_combinations]

    # Find the indices with the maximum sum
    max_sum_indices = max(enumerate(subset_sums), key=lambda x: x[1])[0]

    # Return the indices with the maximum sum
    result_indices = index_combinations[max_sum_indices]

    return result_indices


def nCr(n,r):
    
    f = math.factorial
    return f(n) // f(r) // f(n-r)

def Path_Fraction(a,b,c,Dict,W):
    Term = 0
    if len(a) !=0 and len(b) !=0 and len(c) !=0:
        for item1 in a:
            for item2 in b:
                for item3 in c:
                    Term = Term + (1/W)*(Dict['PM%d' %item1][item2-W-1])*(Dict['PM%d' %item2][item3-2*W-1])
    return Term

def sort_index(List , X):
    x = 0
    INDEX = []
    while(x < X):
        Max = max(List)
        Indx = List.index(Max)
        INDEX.append(Indx)
        List[Indx] = 0
        x = x+1
    return INDEX


def sort_of_clusters(Labels1):
    lists = Labels1
    Index1 = []
    for i in range(len(lists)):
        maxs = 0
        j = 0
        for item in lists:
            if item > maxs:
                maxs = item
                index1 = j
            j = j +1
        lists[index1] = 0
        Index1.append(index1)
    return Index1

def remove_elements_by_index(values, indices):
    """
    Removes elements from 'values' based on the indices given in 'indices'.
    
    Parameters:
        values (list): A list of values.
        indices (list): A list of indices referring to elements to be removed.

    Returns:
        list: The modified 'values' list with specified indices removed.
    """
    # Convert indices to a set to avoid duplicate processing
    index_set = set(indices)
    
    # Create a new list excluding elements at specified indices
    filtered_values = [val for i, val in enumerate(values) if i not in index_set]
    
    return filtered_values


def add_elements_by_index(values, indices):
    """
    Inserts `0` at the specified indices in `values`, maintaining order.

    Parameters:
        values (list): The original list of values.
        indices (list): The list of indices where `0` should be inserted.

    Returns:
        list: A new list with `0` inserted at the specified indices.
    """
    result = []  # Store the final modified list
    value_index = 0  # Track the index in the original values list
    
    for i in range(len(values) + len(indices)):  # Iterate through new length
        if i in indices:  
            result.append(0)  # Insert `0` at specified index
        else:
            result.append(values[value_index])  # Insert original element
            value_index += 1  # Move to the next element in values

    return result

def To_list(List):

    List_ = List.tolist()
    if len(List_)==1:
        output = List_[0]
    else:
        output = List_
    
    return output

def Greedy_(L_M, beta,Omega,fun,Param):
    
    N = len(L_M)

    L = 3
    R_class  = Routing(N,L)
    
    List_Mix = []
    
    r = int(N*np.random.rand(1)[0])
    if r ==N:
        r == N-1
    List_Mix.append(r)
    
    Cap = beta[r]

    while Cap < Omega:
        PDF = []
        for Index in List_Mix:
            Omega_ = beta.copy()
            List0 = To_list(L_M[Index,:])
            List1 = remove_elements_by_index(List0,List_Mix)
            beta0 = remove_elements_by_index(Omega_,List_Mix)

            if not fun== 'Linear':
            
                PDF_List = eval('R_class.'+fun +'(List1,beta0,Param)')
            else:
                PDF_List = [0]*len(List1)
                entery = List1.index(min(List1))
                PDF_List[entery] = 1

            PDF.append(add_elements_by_index(PDF_List,List_Mix))

        Sum = To_list(np.sum(np.matrix(PDF),axis = 0))
        List_Mix.append(Sum.index(max(Sum)))
        Cap += beta[Sum.index(max(Sum))]

    
    return List_Mix
        

def Greedyy(L_M,Max_Omega,beta):
    K = 3
    
    #Max_Omega: Max budget of the adversary
    #K: The minimum mixnodes that should be selected
    #beta id a N*1 vector of the mixnodes capacity
    #L_M N*N matrix including the latencies among the mix-nodes
    C = [] # slectted mixnodes to be corrupted
    N = len(L_M) #number of the mixnodes avaible
    for i in range(N):
        L_M[i,i] = 100000000#To make sure we will not choose one mix-nodes twice
    Budget = 0# take cares of the budget
    Ave_Budget = Max_Omega/K # maxium budget allowed to spend on one mix-nodes corruption
    
    c_0 = round(N*np.random.rand(1)[0])
    if c_0 ==N:
        c_0 = N-1
        while beta[c_0] > Ave_Budget:
            c_0 = round(N*np.random.rand(1)[0])
            if c_0 ==N:
                c_0 = N-1       
    C.append(c_0)
    Budget += beta[c_0]
    
    L_M[:,c_0] = 10000
    
    Index = c_0
    while Budget < Max_Omega:
        
        List0 = To_list(L_M[Index,:])

        List1 = List0.copy()
        
        Index = List1.index(min(List1))
        
        while beta[Index] > Ave_Budget:
            List1[Index] = 1000
            Index = List1.index(min(List1))
            
        C.append(Index)
        L_M[:,Index] = 10000
        Budget += beta[Index]

    return C



def Random(Max_Omega,beta):
    K = 3
    
    Budget = 0# take cares of the budget
    Ave_Budget = Max_Omega/K # maxium budget allowed to spend on one mix-nodes corruption
    N = len(beta)
    C = []
    while Budget < Max_Omega:
        
        Index = round(N*np.random.rand(1)[0])
        if Index ==N:
            Index = N-1
            
        while beta[Index] > Ave_Budget :
            Index = round(N*np.random.rand(1)[0])
            if Index ==N:
                Index = N-1
        
            
        C.append(Index)
        Budget += beta[Index]
        beta[Index] = 10000
       

    return C





def Greedy_For_Fairness(Omega,beta,R_List_,L):

    CNodes = []

    Omega_L = Omega/L

    Cap = 0
    C_List = []
    while Cap < Omega_L:
        Index = beta[0].index(max(beta[0]))

        C_List.append(Index)
        Cap += beta[0][Index]            
        beta[0][Index] = -10000
    CNodes.append(C_List)
    
    
    for l in range(L-1):
        R_List = np.copy(R_List_)
        
        Cap = 0
        C_List = []
        while Cap < Omega_L:
            List_Node = R_List[l][CNodes[l]]

            List_Index = To_list(np.sum(List_Node,axis=0))

            
            Index = List_Index.index(max(List_Index))
            
            R_List[l][:,Index] = -10000
            Cap += beta[l+1][Index]
            beta[l+1][Index] = -10000
            C_List.append(Index)
        CNodes.append(C_List)
            



    return CNodes
    


def Greedy(L_M_,Max_Omega,beta):

    L_M = np.copy(L_M_)
    Cap = 0
    C = []
    
    N = len(L_M)
    
    Index = int(N*np.random.rand(1)[0])
    
    Cap += beta[Index]
    
    L_M[:,Index] = 10000
    C.append(Index)
    
    while Cap < Max_Omega:
        
        
    
        List = To_list(np.sum(L_M[C],axis = 0))

        Index = List.index(min(List))
        
        L_M[:,Index] = 10000
        
        Cap += beta[Index]
        
        C.append(Index)

    return C




