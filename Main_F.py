# -*- coding: utf-8 -*-
"""
Main Functions:
Main_F.py contains a comprehensive set of functions for analyzing OptiMix on stratified topologies, including all necessary functions to compute anonymity and latency metrics.
Additionally, this file includes the main functions required to reproduce the key results of LARMix [29] and LAMP [30].
"""

import numpy as np
import json
import pickle
import math
import simpy
import random
import statistics
import scipy.stats as stats
from scipy.optimize import root


import config
from Message_   import message
from NYM        import MixNet
from GateWay    import GateWay
from Mix_Node_  import Mix
from Routings   import Routing
from Sim_P      import Simulation_P
from Sim        import Simulation
#from Sim_2      import Simulation2
from CLUSTER    import Clustering


from MixNetArrangment      import Mix_Arrangements
from Greedy_LARMIX         import Balanced_Layers
from LARMix_Greedy         import Balanced_Layers

from Message_Genartion_and_mix_net_processing_ import Message_Genartion_and_mix_net_processing
from FCP_Functions_                            import Greedy_, Greedy, Random, Greedy_For_Fairness








def LP_AVE_(L1, L2):
    output = 0
    W,W = np.shape(L1[0])
    for i in range(W):
        for j in range(W):
            for k in range(W):
                for z in range(W):
                    output += (1/W)*(L1[0][i,j]+L1[1][j,k]+L1[2][k,z])*(L2[0][i,j]*L2[1][j,k]*L2[2][k,z])


    return output

def Norm_List(List,term):
    S = np.sum(List)
    return [List[i]*(term/S)for i in range(len(List))]


def Ent_x(List):
    L =[]
    for item in List:
       
        if item!=0:
            L.append(item)
    l = sum(L)
    for i in range(len(L)):
        L[i]=L[i]/l
    ent = 0
    for item in L:
        ent = ent - item*(np.log(item)/np.log(2))
    return ent
def LP_AVE(L1, L2):
    output = 0
    W,W = np.shape(L1[0])
    for i in range(W):
        for j in range(W):
            
            output += (3/W)*(L1[1][i,j])*(L2[1][i,j])


    return output
def rank_elements(input_list):
    # Pair each element with its index
    indexed_list = list(enumerate(input_list))
    
    # Sort the list based on the element values
    sorted_list = sorted(indexed_list, key=lambda x: x[1])
    
    # Initialize a list to store ranks
    ranks = [0] * len(input_list)
    
    # Assign ranks based on sorted position
    for rank, (index, value) in enumerate(sorted_list):
        ranks[index] = rank
    
    return ranks



def SC_Latency(Matrix,Positions,L):
    N = len(Matrix)
    A = np.zeros((N,N))
    W = int(N/L)
    for i in range(N):
        n1 = Positions[i//W][i%W]
        for j in range(N):
            n2 = Positions[j//W][j%W]
            A[i,j] = Matrix[n1,n2]
  
    return A
    


def To_list(List):
    List_ = List.tolist()
    if len(List_)==1:
        output = List_[0]
    else:
        output = List_
    
    return output

  

def EXP(List,Tau):   
    if Tau==1:
        return [1/len(List)]*len(List)
    Min = min(List)
    List_ = [math.exp(-(((Min-item**(1-Tau)))**2)) for item in List]
    List__ = [List_[i]/List[i] for i in range(len(List))]
    sum_ = np.sum(List__)
    dis = [item/sum_ for item in List__]
    return dis 






def equation(x, y, d):
    """ Compute the difference between the left-hand side of the equation and 1. """
    # Avoid division by zero
    if np.any(x * (1 + d) == y):
        return np.inf  # or a large number to represent an invalid result
    sum_term = np.sum(x / (x * (1 + d) - y))
    return sum_term - 1

def solve_x(y, d):
    """ Solve for x given y and d using root with the 'hybr' method. """
    # Initial guess for x
    initial_guess = np.mean(y) + 1.0
    
    # Use root to find the root
    result = root(equation, initial_guess, args=(y, d), method='hybr')
    
    if result.success:
        return result.x[0]
    else:
        raise ValueError("Root finding did not converge")



def Obt(List,Tau):
    List_ = [(1/(item))**((1-Tau)) for item in List]
    x = solve_x(List_, len(List))
    LIST = [(x*(len(List)+1)-(1/(item))**((1-Tau))) for item in List]
    sum_ = np.sum(List_)
    return [x/LIST[i] for i in range(len(LIST))]
    





def equation(X, coefficients, alpha):
    return sum(a * X**(i * alpha) for i, a in enumerate(coefficients, start=1)) - 1

def bisection_method(coefficients, alpha, lower_bound=0, upper_bound=1, tol=1e-7, max_iter=1000):
    if equation(lower_bound, coefficients, alpha) * equation(upper_bound, coefficients, alpha) > 0:
        raise ValueError("The function must have opposite signs at the bounds to apply the bisection method.")
    
    for _ in range(max_iter):
        mid_point = (lower_bound + upper_bound) / 2
        f_mid = equation(mid_point, coefficients, alpha)
        
        if abs(f_mid) < tol:
            return mid_point
        
        if equation(lower_bound, coefficients, alpha) * f_mid < 0:
            upper_bound = mid_point
        else:
            lower_bound = mid_point
    
    raise ValueError("Exceeded maximum iterations. Method did not converge.")

def To_list(matrix):
    """Convert a numpy matrix to a list. If the result is a single row, return it as a flat list."""
    matrix_list = matrix.tolist()
    return matrix_list[0] if len(matrix_list) == 1 else matrix_list

def Fast_Balance(matrix, D_P):
    try:
        (n1,n2) = np.shape(matrix)
    except:
        print(matrix)
    
    """Balance the matrix if the column sums aren't close to 1."""
    list_1 = np.sum(matrix, axis=0)
    if np.allclose(list_1, 1, atol=1/(10**(D_P))):
        return [matrix, True]
    for i in range(n1):
        for j in range(n2):
            if matrix[i,j] ==0:
                matrix[i,j] = 0.0001
    # Normalize the columns by their sums
    new_matrix = matrix / list_1
    
    # Normalize the rows by their sums
    list_2 = np.sum(new_matrix, axis=1).reshape(-1, 1)
    recent_matrix = new_matrix / list_2
    
    return recent_matrix

def Balance_E(matrix, D_P):
    MM = To_list(matrix)
    matrix = np.matrix(MM)
    """Iteratively balance the matrix until the column sums are close to 1."""
    state = False
    new_m = matrix
    IC = 0
    while (not state) and (IC<50):
        
        output = np.copy(new_m)
        new_m = Fast_Balance(new_m, D_P)
        IC = IC+1
        if isinstance(new_m, list):
            state = True
            
    
    return output
    


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









def find_fully_selected_sets(L1, L2):
    # Convert L2 to a set for faster lookup
    L2_set = set(L2)
    
    # Initialize an empty list to store the indices of fully selected sets
    fully_selected_indices = []
    
    # Iterate through L1 and check if each set is fully contained in L2_set
    for index, subset in enumerate(L1):
        if all(element in L2_set for element in subset):
            fully_selected_indices.append(index)
    
    return fully_selected_indices

def List_transpose(List):
    M1 = np.matrix(List)
    M2 = np.transpose(M1)
    L2 = M2.tolist()
    if len(L2)==0:
        output = L2[0]
    else:
        output = L2
    return output


def Med(List):
    N = len(List)

    List_ = []
    for i in range(N):

        List_.append( statistics.median(List[i]))
        
    return List_


def Trans_List(List):
    List_ = []
    for i in range(len(List[0])):
        List__ = []
        for j in range(len(List)):
            List__.append(List[j][i])
        List_.append(List__)
    return List_
            
            


def Entropy(probabilities):
    # Convert to numpy array
    probabilities = np.array(probabilities)
    
    # Remove zero probabilities
    probabilities = probabilities[probabilities != 0]
    
    # Calculate entropy
    entropy_val = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy_val
def probability_greater_than(mean, variance, x):
    # Standard deviation is the square root of variance
    std_dev = variance ** 0.5
    
    # Create a normal distribution with the given mean and standard deviation
    normal_dist = stats.norm(mean, std_dev)
    
    # Calculate P(Z > x)
    probability = 1 - normal_dist.cdf(x)
    
    return probability



def Normal_dist(List,tau):
    mean = np.mean(List)
    Var = np.var(List)
    
    LIST = [0]*(len(List))
    
    for i in range(len(List)):
        LIST[i] = (probability_greater_than(mean,Var,List[i]))**(1-tau)
    Mean = np.sum(LIST)
    Output = [LIST[i]/Mean for i in range(len(List))]
    
    return Output

def top_m(lst_,m):
    lst = lst_.copy()
    # Sort the list in descending order
    sorted_lst = sorted(lst, reverse=True)
    
    # Return the first m elements
    S = sorted_lst[:m]
    Index = []
    for item in S:
        Index.append(lst_.index(item))
    return Index



def Medd(List):
    N = len(List)
    List_ = []
    
    for i in range(N):

        List_.append( statistics.median(List[i]))
        
    return List_

def K_Closets(List,K):
    List_ = List.copy()
    dis = [0]*(len(List))
    Min = 10000000
    Index = []
    for i in range(K):
        index = List_.index(min(List_))
        Index.append(index)
        List_[index] = Min
        
    for item in Index:
        dis[item] = 1/K
    return dis
        
def find_row_permutation(A, B):
    """
    Finds the row permutation mapping from A to B.

    Parameters:
        A (numpy.ndarray): Original matrix (N x M)
        B (numpy.ndarray): Permuted matrix (N x M)

    Returns:
        list: Mapping of rows from A to their positions in B
    """
    A = np.array(A)
    B = np.array(B)

    # Convert each row to a tuple so we can use list index
    A_list = [tuple(row) for row in A]
    B_list = [tuple(row) for row in B]

    # Find indices of A's rows in B
    mapping = [A_list.index(row) for row in B_list]

    return mapping    
    
    
def MAP_to_MAP(L1, L2):
    """
    Computes the permutation from A to C given the permutations from A to B (L1) and B to C (L2).
    
    Parameters:
        L1 (list): Permutation from A to B
        L2 (list): Permutation from B to C
    
    Returns:
        list: Permutation from A to C
    """
    return [L2[i] for i in L1]   


class Carmix(object):
    
    def __init__(self,d,h,W,Targets,run,delay1,delay2,Mix_Threshold,Corrupted_Mix):
        self.d = d
        self.W = W
        self.h = h
        self.c_f = 0.15
        self.N = self.d*self.W*self.h
        self.f = round(self.N*self.c_f)

        self.Targets = Targets
        self.run = run
        self.delay1 = delay1
        self.delay2 = delay2
        self.Corrupted_Mix = Corrupted_Mix
        self.Mix_Threshold = Mix_Threshold
        self.U = self.d
        self.DP = 3
        self.delay3 = 0.2
        self
        
        self.H_N  = self.d
        self.rate = 100
        self.CAP = 10000000000000000000000000000000000000000000000000000000000000000000000000

    
    def Adaptive_Adv(self,R,C):
        Num_chain0 = round((self.N*C)/(self.h*2))
        RR = np.copy(R[1])
        Index_list = To_list(np.sum(R[0],axis = 0))
        LIST = top_m(Index_list, Num_chain0)
        Path_ = np.sum([Index_list[i] for i in LIST])/self.U
        
        LIST2 = []
        
        for j in range(len(LIST)):
            
            Temp_list = To_list(RR[LIST[j],:])
            Next = Temp_list.index(max(Temp_list))
            LIST2.append(Next)
            RR[:,Next] = -10000

        
        
        Path = 0
        for item in LIST:
            for item_ in LIST2:
                x= Index_list[item]*R[1][item,item_]
                Path = Path+x
   
            
        return Path_,Path/self.U
            
   
    

    
    
    def Basic_Adv(self,R,C,Leader_List):
        L = [i for i in range(round(self.N*C))]
        
        Chains = find_fully_selected_sets(Leader_List,L)
        
        C_list = [[]]*self.W
        
        for i in range(self.W):
            C_list[i] = [ j for j in Chains if self.d*(i)<= j <self.d*(1+i)]
            
        Client_preference = To_list(np.sum(R[0],axis =0)/self.U)
        
        
        Path = 0
        for item in C_list[0]:
            for item_ in C_list[1]:
                x= Client_preference[item]*R[1][item,item_-self.d]
                Path = Path+x
        Path_ = 0
        for item in C_list[0]:
            x = Client_preference[item]
            Path_ = Path_+x
            
        return Path_,Path
            

    
    
    def Decod_LONA(self,data0):

        R1 = np.zeros((self.d,self.d))
        R2 = np.copy(R1)
        R3 = np.copy(R1)
        for i in range(self.U):
            R1[i,:] = data0['Client'+str(i+1)]
        
        for k in range(self.W-1):
            for j in range(self.d):
                item = str(k+2)
                if k==0:
                    R2[j,:] = data0['Chain'+str(self.d*k+j+1)]
                else:
                    R3[j,:] = data0['Chain'+str(self.d*k+j+1)]                    

        return R1,R2,R3
            
    

    def Advance_Adv(self,R_1,R_2,CC,b,Leader_List):
        C = round(self.N*CC)
        #R_1 Latency matrix
        #R_2 a list which has one or two routing matrix
        #C is number od corruptted nodes
        #b is brancing 
        #Leader list is the postion of mixnodes in the mixnet
        Latency = R_1.copy()
        R = R_2.copy()
        IC = 0
        L = []
        while IC < C:
            
            r = int(self.N*np.random.rand(1).tolist()[0])
            if r==self.N:
                r = r-1
            while r in L:
                r = int(self.N*np.random.rand(1).tolist()[0])
                if r==self.N:
                    r = r-1  
                    
            L.append(r)
            IC = IC+1
            Index = r
            
            for i in range(b-1):
            
                Temp_list = To_list(Latency[Index,:])
                Index = Temp_list.index(min(Temp_list))
                L.append(Index)
                Latency[:,Index] = 100000
                IC = IC +1
                if not IC < C:
                    break
        

        Chains = find_fully_selected_sets(Leader_List,L)
        
        C_list = [0]*self.W
        
        for i in range(self.W):
            C_list[i] = [ j for j in Chains if self.d*(i)<= j <self.d*(1+i)]
            
        Client_preference = To_list(np.sum(R[0],axis =0)/self.U)
      
        
        Path = 0

        for item in C_list[0]:
            for item_ in C_list[1]:
                x= Client_preference[item]*R[1][item,item_-self.d]
                Path = Path+x
        Path_ = 0
        for item in C_list[0]:
            x = Client_preference[item]
            Path_ = Path_+x
            
        return Path_,Path
    
    def EXP(self,List,Tau):
        LIST = List.copy()
        if Tau==1:
            return [1/len(List)]*len(List)
        Rank = rank_elements(LIST)
        Min = min(List)
        List_ = [2**(-((1-Tau)**2)*Rank[ii]) for ii in range(len(List))]
        List__ = [List_[i]/List[i] for i in range(len(List))]
        sum_ = np.sum(List__)
        dis = [item/sum_ for item in List__]
        return dis 
        

    def GPR(self,List,Tau,G=32  ):
        if Tau==1:
            return [1/len(List)]*len(List)
        
        (Max,Min) = (max(List),min(List))
        Del = (Max-Min)/G
        if Del == 0:
            return [1/len(List)]*len(List)
        
        Index_List = [0]*G
        Map_List = []
        for i,item in enumerate(List):
            Temp_var = (item - Min)//Del
            Temp_var = int(Temp_var)
            if Temp_var == len(Index_List):
                Temp_var -=1
            Index_List[Temp_var] +=1
            Map_List.append(Temp_var)
                
        alpha = bisection_method(Index_List,1)
        
        dis = [ (alpha**(item+1))**(1-Tau) for item in Map_List]
        dis_sum = np.sum(dis)
        return [item/dis_sum for item in dis]

            
            
    
    def LAS(self,List,tau):
    
        
        LIST = [0]*(len(List))
        
        for i in range(len(List)):
            LIST[i] = (1/List[i])**(1-tau)
        Mean = np.sum(LIST)
    
        Output = [LIST[i]/Mean for i in range(len(List))]
        
        return Output

        
    def LAR(self,LIST_,Tau):#We materealize our function for making the trade off
        #In this function just for one sorted distribution
        t = Tau
        A, mapping = self.sort_and_get_mapping(LIST_)
        T = 1-t
    
        B=[]
        D=[]
    
    
        r = 1
        for i in range(len(A)):
            j = i
            J = (j*(1/(t**(r))))**(1-t)
    
            E = math.exp(-1)
            R = E**J
    
            B.append(R)
            A[i] = A[i]**(-T)
    
            g = A[i]*B[i]
    
            D.append(g)
        n=sum(D)
        for l in range(len(D)):
            D[l]=D[l]/n
        restored_list = self.restore_original_list(D, mapping)
    
        return restored_list 
    
    
    def M_Routing(self,fun,Matrix,Tau):
        (n1,n2) = np.shape(Matrix)
        R = np.ones((n1,n2))
        for i in range(n1):
            R[i,:] = fun(To_list(Matrix[i,:]),Tau)
            
        return R

    def Obt(self,List,Tau):
        List_ = [(1/(item))**(0.1*(1-Tau)) for item in List]
        x = solve_x(List_, len(List))
        LIST = [(x*(len(List)+1)-(1/(item))**((1-Tau))) for item in List]
        sum_ = np.sum(List_)
        return [x/LIST[i] for i in range(len(LIST))]  

    def LARMix_Balancing(self,A):
        (n1,n2) = np.shape(A)
        

        LARMix_class = Balanced_Layers(self.DP,'Greedy',n1)
        C = np.copy(A)
        LARMix_class.IMD = C
        
            
        LARMix_class.Iterations()
        
        return LARMix_class.IMD
         
        
               
    
###################Adding Noise###################################
    def Noise(self,R,T):
        (n1,n2) = np.shape(R)
        R1 = np.ones((n1,n2))
        for i in range(len(R)):
            
            List = To_list(R[i,:])
            Max = max(List)
            List_ = [(Max-item)*T for item in List]
            LIST = [List[i]+List_[i] for i in range(len(List))]
            sum_ = np.sum(LIST)
            R1[i,:] = [item/sum_ for item in LIST]
        return R1
            
    

    
    

    def refine(self,List,r):
        output = []
        for i in range(len(List)):
            list_ = [List[i][j] for j in range(len(List[i])) if j in r]
            output.append(list_)
            
        return output
    def balance_check(self,R):
        M,W = np.shape(R)
        ave_R = np.sum(R,axis = 0)/M-1/W
        
        return ave_R
    
    
    def ave_L(self,R,L):
        M,W = np.shape(L)
        return np.sum(L*R)/M
    

    
        
            
    def Greedy(self,R):
        M,W = np.shape(R)
        
        CN = int(self.c*W)
        RR = np.sum(R,axis = 0).tolist()
        
        r = RR.copy()
        
        Index = []
        Max = -10000000
        for i in range(CN):
            Index.append(r.index(max(r)))
            r[r.index(max(r))] = Max
            
        f = 0    
        for j in Index:
            f = f + RR[j]
            
            
        return f/M     
    


    
    def Random(self,R):
        M,W = np.shape(R)
        RR = np.sum(R,axis = 0)
        CN = int(self.c*W)
        Index = []
        
        for i in range(CN):
            r = int(W*np.random.rand(1)[0])
            while r in Index:
                r = int(W*np.random.rand(1)[0])
            Index.append(r)
        f = 0    
        for j in Index:
            f = f + RR[j]
            
            
        return f/M        
 

    def Data_Creation_UNIFORM(self):  
        Data = {}
        for i in range(self.M):
            for j in range(self.M,self.M+self.W):               
                r = np.random.rand
                Data['C'+str(i+1) +'PM'+str(1+j-self.M)] = 0.12*np.random.rand(1)[0]
        return Data   
    
    def Linear(self,L_Matrix,alpha):
        
        R = optimize_pulp(L_Matrix,alpha)
        
        return R

    def sort_and_get_mapping(self,initial_list):
        # Sort the initial list in ascending order and get the sorted indices
        sorted_indices = sorted(range(len(initial_list)), key=lambda x: initial_list[x])
        sorted_list = [initial_list[i] for i in sorted_indices]
    
        # Create a mapping from sorted index to original index
        mapping = {sorted_index: original_index for original_index, sorted_index in enumerate(sorted_indices)}
    
        return sorted_list, mapping
    
    def restore_original_list(self,sorted_list, mapping):
        # Create the original list by mapping each element back to its original position
        original_list = [sorted_list[mapping[i]] for i in range(len(sorted_list))]
        
        return original_list   

    def Ave_latency(self,A,B):
        ave = 0
        for i in range(self.W):
            for j in range(self.W):
                ave = ave + A[i,j]*B[i,j]
        return ave
    def Latency_ave(self,L1,L2):
        latency = 0
        for i in range(len(L1)):
            latency = latency+self.Ave_latency(L1[0],L2[0])
        return latency/self.W
        
            
    
    def T_matrix(self,List):
        
        T = List[0]
        for i in range(len(List)-1):
            T = np.dot(T,List[i+1])
        return T
    def T_entropy(self,List):
        Matrix = self.T_matrix(List)
        (n1,n2) = np.shape(Matrix)
        E = []
        for i in range(n1):
            E.append(Entropy(To_list(Matrix[i,:])))
        
        return np.mean(E)
    
    def Sim(self,List_L,List_R,nn):

        Sim = Simulation([List_L,List_R],self.Targets,self.run,self.delay1,self.delay2,self.d,self.h,self.W,self.U)        

        Latency_Sim,Entropy_Sim = Sim.Simulator(self.Corrupted_Mix,nn)
        
        
        return Latency_Sim,Entropy_Sim
    
    
    
    def Sim2(self,List_L,List_R,nn):
        self.run = 2

        Sim = Simulation2([List_L,List_R],self.Targets,self.run,self.delay3,self.delay2,self.d,self.h,self.W,self.U)        

        Latency_Sim,Entropy_Sim = Sim.Simulator(self.Corrupted_Mix,nn)
        
        
        return Latency_Sim,Entropy_Sim    
    
    
    def data_save(self,Iterations,d,W,h,U):
        self.d = d
        self.W = W
        self.h = h
        self.U = U
        self.N = self.d*self.h*self.W
        data0 = {}
        for i in range(Iterations):
            
            data0['It'+str(i+1)] = To_list(self.Raw_Data())
        with open('D:/OptiMix/Results/data0.json','w') as json_file:
            json.dump(data0,json_file)
            
        
    def Basic_EXP(self,List_Tau,Iterations,state = False):
        Names = ['LAR','LAS','GPR','EXP']
        if state:
            Names = ['GPR'] 
        data0 = {}

        with open('data0.json','r') as json_file:
            Raw_data_ = json.load(json_file)
        
        for It in range(Iterations):
            Matrix = np.matrix(Raw_data_['It'+str(It+1)])
            data2 = self.Random_A(Matrix)

            R1,R2,R3 = self.Decod_LONA(data2)
            L1 = np.copy(R1)
            L2 = np.copy(R2)
            L3 = np.copy(R3)            

            data_ = {}
            for item in Names:
                Temp_data2 = []
                for Tau in List_Tau:
                    
                    if item == 'LAR' and Tau==0:
                        Tau = 0.1
                    
                    r1 = self.M_Routing(eval('self.'+item),R1,Tau)
                    r2 = self.M_Routing(eval('self.'+item),R2,Tau)
                    r3 = self.M_Routing(eval('self.'+item),R3,Tau)                     
                    Temp_data2.append([[L1,L2,L3],[r1,r2,r3]])
                data_[item] = Temp_data2
            data_['List'] = [data2['List']]  
            data_['Mixnodes'] = data2['Mixnodes']
            data0['It'+str(It+1)] = data_
        with open('Results/Basic_data_2.json','wb') as json_file:
            pickle.dump(data0,json_file)            
        return data0
    
    
    def Basic_2_Sim(self,List):
        Mix_Dict = {}
        (W,W) = np.shape(List[0][0])
        
        for i in range(W):
            Mix_Dict['G'+str(i+1)] = [To_list(List[0][0][i,:]),To_list(List[1][0][i,:])]
            
        for j in range(2*W):
            Mix_Dict['PM'+str(j+1)] = [To_list(List[0][j//W+1][j%W,:]),To_list(List[1][j//W+1][j%W,:])]            

        return Mix_Dict
                
    
    
    
    
    
    
    
    
    
    
    
    
        
    def Latency_Entropy(self,T_List,Iterations):
        Names = ['LAR','LAS','GPR','EXP']
        with open('Results/Basic_data_2.json','rb') as json_file:
            data0 = pickle.load(json_file)
        data = {}
        for name in Names:
            data[name] = {}
        for item in Names:
            E_ = []
            L_ = [] 
            E_B_ = []
            L_B_ = []
            E_N_ = []
            E_N_B_ = []                

            for i in range(len(T_List)):
                E = []
                L = [] 
                E_B = []
                L_B = []
                E_N = []
                E_N_B = []                

                
                for It in range(1,Iterations+1):
                    #Latency of LONA for one and two wings setting
                    L.append(LP_AVE(data0['It'+str(It)][item][i][0],data0['It'+str(It)][item][i][1]))

                    
                    #Entropy of LONA for one and two wings setting
                    E.append(self.T_entropy(data0['It'+str(It)][item][i][1]))

                    #Balance the R1 AND R2 for LONA
                    B1 = Balance_E(data0['It'+str(It)][item][i][1][0],self.DP)
                    B2 = Balance_E(data0['It'+str(It)][item][i][1][1],self.DP)
                    B3 = Balance_E(data0['It'+str(It)][item][i][1][2],self.DP)                    
                    #Latency of LONA for one and two eings setting
                    L_B.append(LP_AVE(data0['It'+str(It)][item][i][0],[B1,B2,B3]))
                    E_B.append(self.T_entropy([B1,B2,B3])) 
                    

                    
                    #Adding noise to the rouitng distributions for LONA
                    N1 = self.Noise(data0['It'+str(It)][item][i][1][0],0.02)
                    N2 = self.Noise(data0['It'+str(It)][item][i][1][1],0.02)  
                    N3 = self.Noise(data0['It'+str(It)][item][i][1][2],0.02)                   


                    E_N.append(self.T_entropy([N1,N2,N3]))

                    
                    #Adding noise to the rouitng distributions for LONA
                    NB1 = self.Noise(B1,0.5)
                    NB2 = self.Noise(B2,0.5)     
                    NB3 = self.Noise(B3,0.5)                    

                
                E_.append(np.mean(E))
                L_.append(np.mean(L)) 
                E_B_.append(np.mean(E_B))
                L_B_.append(np.mean(L_B))
                E_N_.append(np.mean(E_N))
                
            data[item]['E']     = E_                    
            data[item]['L']     = L_                  
            data[item]['E_B']   = E_B_                    
            data[item]['L_B']   = L_B_
            data[item]['E_N']   = E_N_
            data[item]['E_N_B'] = E_N_B
        with open('Results/LE_data.pkl','wb') as json_file:
            pickle.dump(data,json_file)
  
    
    
    
    
    
    
    def Noise_Latency_Entropy(self,N_List,Iterations):
        Names = ['LAR','LAS','GPR','EXP']
        with open('D:/Cascades/Results/Basic_data.json','rb') as json_file:
            data0 = pickle.load(json_file)
        data = {}

        for name in Names:
            data[name] = {}
        for item in Names:

            E_W_LONA_Noise_ = []
            E_WW_LONA_Noise_ = []  
            E_W_LONA_Noise_B_ = []
            E_WW_LONA_Noise_B_ = []                

            for i in range(len(N_List)):
                
                E_W_LONA_Noise = []
                E_WW_LONA_Noise = []
                E_W_LONA_Noise_B = []
                E_WW_LONA_Noise_B = []                


                for It in range(1,Iterations+1):

                    
                    #Entropy of LONA for one and two wings setting
                    R1 = data0['It'+str(It)][item][1][3][0][0]
                    R2 = data0['It'+str(It)][item][1][3][0][1]

                    B1 = Balance_E(data0['It'+str(It)][item][1][3][0][0],self.DP)
                    B2 = Balance_E(data0['It'+str(It)][item][1][3][0][1],self.DP)


                    
                    #Adding noise to the rouitng distributions for LONA
                    N1 = self.Noise(R1,N_List[i])
                    N2 = self.Noise(R2,N_List[i])                    

                    R1_N = self.T_entropy(N1)
                    R2_N = self.T_entropy(self.T_matrix([N1,N2])) 
                    E_W_LONA_Noise.append(R1_N)
                    E_WW_LONA_Noise.append(R2_N)
                    
                    
                    #Adding noise to the rouitng distributions for LONA
                    NN1 = self.Noise(B1,N_List[i])
                    NN2 = self.Noise(B2,N_List[i])                    

                    R1_NN = self.T_entropy(NN1)
                    R2_NN = self.T_entropy(self.T_matrix([NN1,NN2])) 
                    E_W_LONA_Noise_B.append(R1_NN)
                    E_WW_LONA_Noise_B.append(R2_NN)


                
  
                E_W_LONA_Noise_.append(np.mean(E_W_LONA_Noise))
                E_WW_LONA_Noise_.append(np.mean(E_WW_LONA_Noise))
                E_W_LONA_Noise_B_.append(np.mean(E_W_LONA_Noise_B))
                E_WW_LONA_Noise_B_.append(np.mean(E_WW_LONA_Noise_B))                
                


            data[item]['E_W_LONA_Noise'] = E_W_LONA_Noise_
            data[item]['E_WW_LONA_Noise'] = E_WW_LONA_Noise_ 

            data[item]['E_W_LONA_Noise_B'] = E_W_LONA_Noise_B_
            data[item]['E_WW_LONA_Noise_B'] = E_WW_LONA_Noise_B_
            

        
        return data        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    def Adversary_Analysis(self,T_List,Iterations):
        
        Names = ['Random','Adaptive']

        with open('D:/Cascades/Results/Basic_data.json','rb') as json_file:
            data0 = pickle.load(json_file)        
        data = {}

        for name in Names:
            data[name] = {}
        for item in Names:
            FCP_W_LONA_ = []
            FCP_WW_LONA_ = []
            FCP_W_Random_ = []
            FCP_WW_Random_ = []
            
            FCP_W_LONA_B_ = []
            FCP_WW_LONA_B_ = []  
            FCP_W_LONA_Noise_ = []
            FCP_WW_LONA_Noise_ = []
            FCP_W_LONA_Noise_B_ = []
            FCP_WW_LONA_Noise_B_ = []            
            for i in range(len(T_List)):
                FCP_W_LONA = []
                FCP_WW_LONA = []
                FCP_W_Random = []
                FCP_WW_Random = []
                FCP_W_LONA_B = []
                FCP_WW_LONA_B = []
                FCP_W_LONA_Noise = []
                FCP_WW_LONA_Noise = []
                FCP_W_LONA_Noise_B = []
                FCP_WW_LONA_Noise_B = []                
                for It in range(1,Iterations+1):
                    Leader_List = data0['It'+str(It)]['List'][0]
                    Latency_Matrix = data0['It'+str(It)]['Mixnodes'][0]
                    Leader_List1 = data0['It'+str(It)]['List'][1]
                    Latency_Matrix1 = data0['It'+str(It)]['Mixnodes'][1]                    
                    

                    #FCP of LONA for one and two wings setting
                      

                    R_List = [data0['It'+str(It)]['GPR'][1][i][0][0],data0['It'+str(It)]['GPR'][1][i][0][1]]

                    if item == 'Advance':
                        Path,Path_ = self.Advance_Adv(Latency_Matrix, R_List, self.c_f, round((self.N*self.c_f)/2*self.h), Leader_List)                    
                    elif item=='Greedy':
                        Path,Path_ = self.Advance_Adv(Latency_Matrix, R_List, self.c_f,round((self.N*self.c_f)/self.h), Leader_List)                         
                    elif item=='Random':
                        
                        Path,Path_ = self.Basic_Adv(R_List, self.c_f, Leader_List) 
                    elif item=='Adaptive':
                        Path,Path_ = self.Adaptive_Adv(R_List, self.c_f)

                    FCP_W_LONA.append(Path)
                    FCP_WW_LONA.append(Path_)
                    
                     #FCP of Random for one and two wings setting
                      

                    R_List = [data0['It'+str(It)]['GPR'][1][i][1][0],data0['It'+str(It)]['GPR'][1][i][1][1]]

                    if item == 'Advance':
                        Path,Path_  = self.Advance_Adv(Latency_Matrix1, R_List, self.c_f, round((self.N*self.c_f)/2*self.h), Leader_List1)                    
                    elif item=='Greedy':
                        Path,Path_  = self.Advance_Adv(Latency_Matrix1, R_List, self.c_f,round((self.N*self.c_f)/self.h), Leader_List1)                         
                    elif item=='Random':
                        
                        Path,Path_  = self.Basic_Adv(R_List, self.c_f, Leader_List1)                   
                    elif item=='Adaptive':
                        Path,Path_ = self.Adaptive_Adv(R_List, self.c_f)                    
                    FCP_W_Random.append(Path)
                    FCP_WW_Random.append(Path_)
                    #Balance the R1 AND R2 for LONA to be used for FCP
                    B1 = Balance_E(data0['It'+str(It)]['GPR'][1][i][0][0],self.DP)
                    B2 = Balance_E(data0['It'+str(It)]['GPR'][1][i][0][1],self.DP)
                    #FCP of LONA for one and two wings setting for B

                    R_List = [B1,B2]

                    if item == 'Advance':
                        Path,Path_  = self.Advance_Adv(Latency_Matrix, R_List, self.c_f, round((self.N*self.c_f)/2*self.h), Leader_List)                    
                    elif item=='Greedy':
                        Path,Path_  = self.Advance_Adv(Latency_Matrix, R_List, self.c_f,round((self.N*self.c_f)/self.h), Leader_List)                         
                    elif item=='Random':
                        Path,Path_  = self.Basic_Adv(R_List, self.c_f, Leader_List)
                    elif item=='Adaptive':
                        Path,Path_ = self.Adaptive_Adv(R_List, self.c_f)                        
                    FCP_W_LONA_B.append(Path)
                    FCP_WW_LONA_B.append(Path_)
                 
                    
                    
                    #Adding noise to the rouitng distributions for LONA to compute FCP
                    N1 = self.Noise(data0['It'+str(It)]['GPR'][1][i][0][0],0.5)
                    N2 = self.Noise(data0['It'+str(It)]['GPR'][1][i][0][1],0.5)                    



                    #FCP of LONA for one and two wings setting
                    R_List = [N1,N2]

                    if item == 'Advance':
                        Path,Path_  = self.Advance_Adv(Latency_Matrix, R_List, self.c_f, round((self.N*self.c_f)/2*self.h), Leader_List)                    
                    elif item=='Greedy':
                        Path,Path_  = self.Advance_Adv(Latency_Matrix, R_List, self.c_f,round((self.N*self.c_f)/self.h), Leader_List)                         
                    elif item=='Random':
                        Path,Path_  = self.Basic_Adv(R_List, self.c_f, Leader_List)
                    elif item=='Adaptive':
                        Path,Path_ = self.Adaptive_Adv(R_List, self.c_f)                        
                    FCP_W_LONA_Noise.append(Path)
                    FCP_WW_LONA_Noise.append(Path_)
                    
                    
                
                    #Adding noise to the rouitng distributions for LONA to compute FCP
                    NN1 = self.Noise(B1,0.5)
                    NN2 = self.Noise(B2,0.5)                    



                    #FCP of LONA for one and two wings setting
                    R_List = [NN1,NN2]

                    if item == 'Advance':
                        Path,Path_  = self.Advance_Adv(Latency_Matrix, R_List, self.c_f, round((self.N*self.c_f)/2*self.h), Leader_List)                    
                    elif item=='Greedy':
                        Path,Path_  = self.Advance_Adv(Latency_Matrix, R_List, self.c_f,round((self.N*self.c_f)/self.h), Leader_List)                         
                    elif item=='Random':
                        Path,Path_  = self.Basic_Adv(R_List, self.c_f, Leader_List)
                    elif item=='Adaptive':
                        Path,Path_ = self.Adaptive_Adv(R_List, self.c_f)                        
                    FCP_W_LONA_Noise_B.append(Path)
                    FCP_WW_LONA_Noise_B.append(Path_)                    
                    
                FCP_W_LONA_.append(np.mean(FCP_W_LONA))
                FCP_WW_LONA_.append(np.mean(FCP_WW_LONA))
                FCP_W_Random_.append(np.mean(FCP_W_Random))
                FCP_WW_Random_.append(np.mean(FCP_WW_Random))

                FCP_W_LONA_B_.append(np.mean(FCP_W_LONA_B))
                FCP_WW_LONA_B_.append(np.mean(FCP_WW_LONA_B))

                FCP_W_LONA_Noise_.append(np.mean(FCP_W_LONA_Noise))
                FCP_WW_LONA_Noise_.append(np.mean(FCP_WW_LONA_Noise))


                FCP_W_LONA_Noise_B_.append(np.mean(FCP_W_LONA_Noise_B))
                FCP_WW_LONA_Noise_B_.append(np.mean(FCP_WW_LONA_Noise_B))
            data[item]['FCP_W_LONA'] = FCP_W_LONA_
            data[item]['FCP_WW_LONA'] = FCP_WW_LONA_                    
            data[item]['FCP_W_Random'] = FCP_W_Random_                    
            data[item]['FCP_WW_Random'] = FCP_WW_Random_                    
                
                    
                    
            data[item]['FCP_W_LONA_B'] = FCP_W_LONA_B_
            data[item]['FCP_WW_LONA_B'] = FCP_WW_LONA_B_                    
                   
  
            data[item]['FCP_W_LONA_Noise'] = FCP_W_LONA_Noise_
            data[item]['FCP_WW_LONA_Noise'] = FCP_WW_LONA_Noise_ 

            data[item]['FCP_W_LONA_Noise_B'] = FCP_W_LONA_Noise_B_
            data[item]['FCP_WW_LONA_Noise_B'] = FCP_WW_LONA_Noise_B_        
        return data     
                    
        
                
                
                
                
                
                
                
                                    
                
                
            
    


    def Adversary_Budget(self,B_List,Iterations):
        
        Names = ['Advance','Greedy','Random','Adaptive']

        with open('D:/Cascades/Results/Basic_data.json','rb') as json_file:
            data0 = pickle.load(json_file)          

        data = {}

        for name in Names:
            data[name] = {}
        for item in Names:
            FCP_W_LONA_ = []
            FCP_WW_LONA_ = []
            FCP_W_Random_ = []
            FCP_WW_Random_ = []
            
            FCP_W_LONA_B_ = []
            FCP_WW_LONA_B_ = []  
            FCP_W_LONA_Noise_ = []
            FCP_WW_LONA_Noise_ = []
            FCP_W_LONA_Noise_B_ = []
            FCP_WW_LONA_Noise_B_ = []            
            for ii in range(len(B_List)):
                self.c_f = B_List[ii]
                FCP_W_LONA = []
                FCP_WW_LONA = []
                FCP_W_Random = []
                FCP_WW_Random = []
                FCP_W_LONA_B = []
                FCP_WW_LONA_B = []
                FCP_W_LONA_Noise = []
                FCP_WW_LONA_Noise = []
                FCP_W_LONA_Noise_B = []
                FCP_WW_LONA_Noise_B = []                
                for It in range(1,Iterations+1):
                    Leader_List = data0['It'+str(It)]['List'][0]
                    Latency_Matrix = data0['It'+str(It)]['Mixnodes'][0]
                    Leader_List1 = data0['It'+str(It)]['List'][1]
                    Latency_Matrix1 = data0['It'+str(It)]['Mixnodes'][1]                    
                    
                    #FCP of LONA for one and two wings setting
                      

                    R_List = [data0['It'+str(It)]['GPR'][1][3][0][0],data0['It'+str(It)]['GPR'][1][3][0][1]]

                    if item == 'Advance':
                        Path,Path_ = self.Advance_Adv(Latency_Matrix, R_List, self.c_f, round((self.N*self.c_f)/2*self.h), Leader_List)                    
                    elif item=='Greedy':
                        Path,Path_ = self.Advance_Adv(Latency_Matrix, R_List, self.c_f,round((self.N*self.c_f)/self.h), Leader_List)                         
                    elif item=='Random':
                        
                        Path,Path_ = self.Basic_Adv(R_List, self.c_f, Leader_List) 
                    elif item=='Adaptive':
                        Path,Path_ = self.Adaptive_Adv(R_List, self.c_f)

                    FCP_W_LONA.append(Path)
                    FCP_WW_LONA.append(Path_)
                    
                     #FCP of Random for one and two wings setting
                      

                    R_List = [data0['It'+str(It)]['GPR'][1][3][1][0],data0['It'+str(It)]['GPR'][1][3][1][1]]

                    if item == 'Advance':
                        Path,Path_  = self.Advance_Adv(Latency_Matrix1, R_List, self.c_f, round((self.N*self.c_f)/2*self.h), Leader_List1)                    
                    elif item=='Greedy':
                        Path,Path_  = self.Advance_Adv(Latency_Matrix1, R_List, self.c_f,round((self.N*self.c_f)/self.h), Leader_List1)                         
                    elif item=='Random':
                        
                        Path,Path_  = self.Basic_Adv(R_List, self.c_f, Leader_List1)                   
                    elif item=='Adaptive':
                        Path,Path_ = self.Adaptive_Adv(R_List, self.c_f)                    
                    FCP_W_Random.append(Path)
                    FCP_WW_Random.append(Path_)
                    #Balance the R1 AND R2 for LONA to be used for FCP
                    B1 = Balance_E(data0['It'+str(It)]['GPR'][1][3][0][0],self.DP)
                    B2 = Balance_E(data0['It'+str(It)]['GPR'][1][3][0][1],self.DP)
                    #FCP of LONA for one and two wings setting for B

                    R_List = [B1,B2]

                    if item == 'Advance':
                        Path,Path_  = self.Advance_Adv(Latency_Matrix, R_List, self.c_f, round((self.N*self.c_f)/2*self.h), Leader_List)                    
                    elif item=='Greedy':
                        Path,Path_  = self.Advance_Adv(Latency_Matrix, R_List, self.c_f,round((self.N*self.c_f)/self.h), Leader_List)                         
                    elif item=='Random':
                        Path,Path_  = self.Basic_Adv(R_List, self.c_f, Leader_List)
                    elif item=='Adaptive':
                        Path,Path_ = self.Adaptive_Adv(R_List, self.c_f)                        
                    FCP_W_LONA_B.append(Path)
                    FCP_WW_LONA_B.append(Path_)
                 
                    
                    
                    #Adding noise to the rouitng distributions for LONA to compute FCP
                    N1 = self.Noise(data0['It'+str(It)]['GPR'][1][3][0][0],0.5)
                    N2 = self.Noise(data0['It'+str(It)]['GPR'][1][3][0][1],0.5)                    



                    #FCP of LONA for one and two wings setting
                    R_List = [N1,N2]

                    if item == 'Advance':
                        Path,Path_  = self.Advance_Adv(Latency_Matrix, R_List, self.c_f, round((self.N*self.c_f)/2*self.h), Leader_List)                    
                    elif item=='Greedy':
                        Path,Path_  = self.Advance_Adv(Latency_Matrix, R_List, self.c_f,round((self.N*self.c_f)/self.h), Leader_List)                         
                    elif item=='Random':
                        Path,Path_  = self.Basic_Adv(R_List, self.c_f, Leader_List)
                    elif item=='Adaptive':
                        Path,Path_ = self.Adaptive_Adv(R_List, self.c_f)                        
                    FCP_W_LONA_Noise.append(Path)
                    FCP_WW_LONA_Noise.append(Path_)
                    
                    
                
                    #Adding noise to the rouitng distributions for LONA to compute FCP
                    NN1 = self.Noise(B1,0.5)
                    NN2 = self.Noise(B2,0.5)                    



                    #FCP of LONA for one and two wings setting
                    R_List = [NN1,NN2]

                    if item == 'Advance':
                        Path,Path_  = self.Advance_Adv(Latency_Matrix, R_List, self.c_f, round((self.N*self.c_f)/2*self.h), Leader_List)                    
                    elif item=='Greedy':
                        Path,Path_  = self.Advance_Adv(Latency_Matrix, R_List, self.c_f,round((self.N*self.c_f)/self.h), Leader_List)                         
                    elif item=='Random':
                        Path,Path_  = self.Basic_Adv(R_List, self.c_f, Leader_List)
                    elif item=='Adaptive':
                        Path,Path_ = self.Adaptive_Adv(R_List, self.c_f)                        
                    FCP_W_LONA_Noise_B.append(Path)
                    FCP_WW_LONA_Noise_B.append(Path_)                    
                    
                FCP_W_LONA_.append(np.mean(FCP_W_LONA))
                FCP_WW_LONA_.append(np.mean(FCP_WW_LONA))
                FCP_W_Random_.append(np.mean(FCP_W_Random))
                FCP_WW_Random_.append(np.mean(FCP_WW_Random))

                FCP_W_LONA_B_.append(np.mean(FCP_W_LONA_B))
                FCP_WW_LONA_B_.append(np.mean(FCP_WW_LONA_B))

                FCP_W_LONA_Noise_.append(np.mean(FCP_W_LONA_Noise))
                FCP_WW_LONA_Noise_.append(np.mean(FCP_WW_LONA_Noise))


                FCP_W_LONA_Noise_B_.append(np.mean(FCP_W_LONA_Noise_B))
                FCP_WW_LONA_Noise_B_.append(np.mean(FCP_WW_LONA_Noise_B))
            data[item]['FCP_W_LONA'] = FCP_W_LONA_
            data[item]['FCP_WW_LONA'] = FCP_WW_LONA_                    
            data[item]['FCP_W_Random'] = FCP_W_Random_                    
            data[item]['FCP_WW_Random'] = FCP_WW_Random_                    
                
                    
                    
            data[item]['FCP_W_LONA_B'] = FCP_W_LONA_B_
            data[item]['FCP_WW_LONA_B'] = FCP_WW_LONA_B_                    
                   
  
            data[item]['FCP_W_LONA_Noise'] = FCP_W_LONA_Noise_
            data[item]['FCP_WW_LONA_Noise'] = FCP_WW_LONA_Noise_ 

            data[item]['FCP_W_LONA_Noise_B'] = FCP_W_LONA_Noise_B_
            data[item]['FCP_WW_LONA_Noise_B'] = FCP_WW_LONA_Noise_B_        
        return data     
    
    
    def Baseline_Sim(self,T_List,nn,Iterations):
        Names = ['GPR']
        data = {}

        with open('D:/Cascades/Results/Basic_data.json','rb') as json_file:
            data0 = pickle.load(json_file)          
        for name in Names:
            data[name] = {}
        for item in Names:

            E_W_LONA_ = []
            E_WW_LONA_ = []
            E_W_Random_ = []
            E_WW_Random_ = []
            
            L_W_LONA_ = []
            L_WW_LONA_ = []
            L_W_Random_ = []
            L_WW_Random_ = [] 
            E_W_LONA_B_ = []
            E_WW_LONA_B_ = []
            L_W_LONA_B_ = []
            L_WW_LONA_B_ = []  

            for i in range(len(T_List)):

                E_W_LONA = []
                E_WW_LONA = []
                E_W_Random = []
                E_WW_Random = []
                
                L_W_LONA = []
                L_WW_LONA = []
                L_W_Random = []
                L_WW_Random = []  

                E_W_LONA_B = []
                E_WW_LONA_B = []
                L_W_LONA_B = []
                L_WW_LONA_B = []

                for It in range(1,Iterations+1):
                    #Latency of LONA for one and two wings setting
                    R1 = [data0['It'+str(It)][item][1][i][0][0],data0['It'+str(It)][item][1][i][0][1]]
                    L1 = [data0['It'+str(It)][item][0][i][0][0],data0['It'+str(It)][item][0][i][0][1]]
    
                    R2 = [data0['It'+str(It)][item][1][i][1][0],data0['It'+str(It)][item][1][i][1][1]]
                    L2 = [data0['It'+str(It)][item][0][i][1][0],data0['It'+str(It)][item][0][i][1][1]]
    
    
                    Ave_L, Ave_E = self.Sim(L1, R1, nn)
                    Ave_L_, Ave_E_ = self.Sim(L2, R2, nn)    
    

                    L_W_LONA.append(Ave_L[0])
                    L_WW_LONA.append(Ave_L[1])
                    L_W_Random.append(Ave_L_[0])
                    L_WW_Random.append(Ave_L_[1])
                    
                    #Entropy of LONA for one and two wings setting

                    E_W_LONA.append(Ave_E[0])
                    E_WW_LONA.append(Ave_E[1])
                    E_W_Random.append(Ave_E_[0])
                    E_WW_Random.append(Ave_E_[1])
                    #Balance the R1 AND R2 for LONA
                    B1 = Balance_E(R1[0],self.DP)
                    B2 = Balance_E(R1[1],self.DP)
                    Ave_L_B, Ave_E_B = self.Sim(L1,[B1,B2], nn) 

                    
                    L_W_LONA_B.append(Ave_L_B[0])
                    L_WW_LONA_B.append(Ave_L_B[1])
                    E_W_LONA_B.append(Ave_E_B[0])
                    E_WW_LONA_B.append(Ave_E_B[1])                    
                    
                    

                    
                E_W_LONA_.append(E_W_LONA)
                E_WW_LONA_.append(E_WW_LONA)
                E_W_Random_.append(E_W_Random)
                E_WW_Random_.append(E_WW_Random)
                
                L_W_LONA_.append(L_W_LONA)
                L_WW_LONA_.append(L_WW_LONA)
                L_W_Random_.append(L_W_Random)
                L_WW_Random_.append(L_WW_Random)
                E_W_LONA_B_.append(E_W_LONA_B)
                E_WW_LONA_B_.append(E_WW_LONA_B)
                L_W_LONA_B_.append(L_W_LONA_B)
                L_WW_LONA_B_.append(L_WW_LONA_B)

            data[item]['E_W_LONA'] = E_W_LONA_
            data[item]['E_WW_LONA'] = E_WW_LONA_                    
            data[item]['E_W_Random'] = E_W_Random_                    
            data[item]['E_WW_Random'] = E_WW_Random_                    

            data[item]['L_W_LONA'] = L_W_LONA_
            data[item]['L_WW_LONA'] = L_WW_LONA_                    
            data[item]['L_W_Random'] = L_W_Random_                    
            data[item]['L_WW_Random'] = L_WW_Random_                     
                    
                    
            data[item]['E_W_LONA_B'] = E_W_LONA_B_
            data[item]['E_WW_LONA_B'] = E_WW_LONA_B_                    
            data[item]['L_W_LONA_B'] = L_W_LONA_B_
            data[item]['L_WW_LONA_B'] = L_WW_LONA_B_                    
  

        
        return data                             
    

    
    
    
    def Baseline_Sim_T(self,T_List,nn,Iterations):

        Names = ['GPR']

        data = {}

        with open('D:/Cascades/Results/Basic_data.json','rb') as json_file:
            data0 = pickle.load(json_file)         
        for name in Names:
            data[name] = {}
        for item in Names:
            E_W_LONA_ = []
            E_WW_LONA_ = []
            E_W_Random_ = []
            E_WW_Random_ = []
            
            L_W_LONA_ = []
            L_WW_LONA_ = []
            L_W_Random_ = []
            L_WW_Random_ = [] 
            E_W_LONA_B_ = []
            E_WW_LONA_B_ = []
            L_W_LONA_B_ = []
            L_WW_LONA_B_ = []  

            for i in range(len(T_List)):

                E_W_LONA = []
                E_WW_LONA = []
                E_W_Random = []
                E_WW_Random = []
                
                L_W_LONA = []
                L_WW_LONA = []
                L_W_Random = []
                L_WW_Random = []  

                E_W_LONA_B = []
                E_WW_LONA_B = []
                L_W_LONA_B = []
                L_WW_LONA_B = []

                
                for It in range(1,Iterations+1):
                    #Latency of LONA for one and two wings setting
                    R1 = [data0['It'+str(It)][item][1][i][0][0],data0['It'+str(It)][item][1][i][0][1]]
                    L1 = [data0['It'+str(It)][item][0][i][0][0],data0['It'+str(It)][item][0][i][0][1]]
    
                    R2 = [data0['It'+str(It)][item][1][i][1][0],data0['It'+str(It)][item][1][i][1][1]]
                    L2 = [data0['It'+str(It)][item][0][i][1][0],data0['It'+str(It)][item][0][i][1][1]]
    
    
                    Ave_L, Ave_E = self.Sim2(L1, R1, nn)
                    Ave_L_, Ave_E_ = self.Sim2(L2, R2, nn)    
    

                    L_W_LONA.append(Ave_L[0])
                    L_WW_LONA.append(Ave_L[1])
                    L_W_Random.append(Ave_L_[0])
                    L_WW_Random.append(Ave_L_[1])
                    
                    #Entropy of LONA for one and two wings setting

                    E_W_LONA.append(Ave_E[0])
                    E_WW_LONA.append(Ave_E[1])
                    E_W_Random.append(Ave_E_[0])
                    E_WW_Random.append(Ave_E_[1])
                    #Balance the R1 AND R2 for LONA
                    B1 = Balance_E(R1[0],self.DP)
                    B2 = Balance_E(R1[1],self.DP)
                    Ave_L_B, Ave_E_B = self.Sim2(L1,[B1,B2], nn) 

                    
                    L_W_LONA_B.append(Ave_L_B[0])
                    L_WW_LONA_B.append(Ave_L_B[1])
                    E_W_LONA_B.append(Ave_E_B[0])
                    E_WW_LONA_B.append(Ave_E_B[1])                    
                    
                    

                    
                E_W_LONA_.append(E_W_LONA)
                E_WW_LONA_.append(E_WW_LONA)
                E_W_Random_.append(E_W_Random)
                E_WW_Random_.append(E_WW_Random)
                
                L_W_LONA_.append(L_W_LONA)
                L_WW_LONA_.append(L_WW_LONA)
                L_W_Random_.append(L_W_Random)
                L_WW_Random_.append(L_WW_Random)
                E_W_LONA_B_.append(E_W_LONA_B)
                E_WW_LONA_B_.append(E_WW_LONA_B)
                L_W_LONA_B_.append(L_W_LONA_B)
                L_WW_LONA_B_.append(L_WW_LONA_B)  

            data[item]['E_W_LONA'] = E_W_LONA_
            data[item]['E_WW_LONA'] = E_WW_LONA_                    
            data[item]['E_W_Random'] = E_W_Random_                    
            data[item]['E_WW_Random'] = E_WW_Random_                    

            data[item]['L_W_LONA'] = L_W_LONA_
            data[item]['L_WW_LONA'] = L_WW_LONA_                    
            data[item]['L_W_Random'] = L_W_Random_                    
            data[item]['L_WW_Random'] = L_WW_Random_                     
                    
                    
            data[item]['E_W_LONA_B'] = E_W_LONA_B_
            data[item]['E_WW_LONA_B'] = E_WW_LONA_B_                    
            data[item]['L_W_LONA_B'] = L_W_LONA_B_
            data[item]['L_WW_LONA_B'] = L_WW_LONA_B_                    
  

        
        return data                             
    
    
 
    def Network_Size(self,d_List,Iterations,nn):
        Names = ['GPR']

        data = {}

        data = {}
        for d in (d_List):
            data[str(d)] = {}
        for d in d_List:
            self.d = d
            self.U = 2*self.d
            data0 = self.Basic_EXP([0.6],Iterations,True)
            E_W_LONA_ = []
            E_WW_LONA_ = []
            E_W_Random_ = []
            E_WW_Random_ = []
            
            L_W_LONA_ = []
            L_WW_LONA_ = []
            L_W_Random_ = []
            L_WW_Random_ = [] 

            SE_W_LONA_ = []
            SE_WW_LONA_ = []
            SE_W_Random_ = []
            SE_WW_Random_ = []
            
            SL_W_LONA_ = []
            SL_WW_LONA_ = []
            SL_W_Random_ = []
            SL_WW_Random_ = []             

            for i in range(1):
                E_W_LONA = []
                E_WW_LONA = []
                E_W_Random = []
                E_WW_Random = []
                
                L_W_LONA = []
                L_WW_LONA = []
                L_W_Random = []
                L_WW_Random = []  


                SE_W_LONA = []
                SE_WW_LONA = []
                SE_W_Random = []
                SE_WW_Random = []
                
                SL_W_LONA = []
                SL_WW_LONA = []
                SL_W_Random = []
                SL_WW_Random = []  

                for It in range(1,len(data0)+1):
                    #Latency of LONA for one and two wings setting
                    L1 = self.ave_L(data0['It'+str(It)]['GPR'][1][i][0][0],data0['It'+str(It)]['GPR'][0][i][0][0])
                    L2 = self.ave_L(data0['It'+str(It)]['GPR'][1][i][0][1],data0['It'+str(It)]['GPR'][0][i][0][1]) 
                    #Latency of Random for one and two wings setting
                    L_1 = self.ave_L(data0['It'+str(It)]['GPR'][1][i][1][0],data0['It'+str(It)]['GPR'][0][i][1][0])
                    L_2 = self.ave_L(data0['It'+str(It)]['GPR'][1][i][1][1],data0['It'+str(It)]['GPR'][0][i][1][1])
                    
                    L_W_LONA.append(L1)
                    L_WW_LONA.append(L1+L2)
                    L_W_Random.append(L_1)
                    L_WW_Random.append(L_1+L_2)
                    
                    #Entropy of LONA for one and two wings setting
                    R1 = self.T_entropy(data0['It'+str(It)]['GPR'][1][i][0][0])
                    R2 = self.T_entropy(self.T_matrix([data0['It'+str(It)]['GPR'][1][i][0][0],data0['It'+str(It)]['GPR'][1][i][0][1]]))
                    #Entropy of Random for one and two wings setting
                    R_1 = self.T_entropy(data0['It'+str(It)]['GPR'][1][i][1][0])
                    R_2 = self.T_entropy(self.T_matrix([data0['It'+str(It)]['GPR'][1][i][1][0],data0['It'+str(It)]['GPR'][1][i][1][1]]))                   

                    E_W_LONA.append(R1)
                    E_WW_LONA.append(R2)
                    E_W_Random.append(R_1)
                    E_WW_Random.append(R_2)
                    #Balance the R1 AND R2 for LONA
                    R1 = [data0['It'+str(It)]['GPR'][1][i][0][0],data0['It'+str(It)]['GPR'][1][i][0][1]]
                    L1 = [data0['It'+str(It)]['GPR'][0][i][0][0],data0['It'+str(It)]['GPR'][0][i][0][1]]
    
                    R2 = [data0['It'+str(It)]['GPR'][1][i][1][0],data0['It'+str(It)]['GPR'][1][i][1][1]]
                    L2 = [data0['It'+str(It)]['GPR'][0][i][1][0],data0['It'+str(It)]['GPR'][0][i][1][1]]
    
    
                    Ave_L, Ave_E = self.Sim(L1, R1, nn)
                    Ave_L_, Ave_E_ = self.Sim(L2, R2, nn)    
    

                    SL_W_LONA.append(Ave_L[0])
                    SL_WW_LONA.append(Ave_L[1])
                    SL_W_Random.append(Ave_L_[0])
                    SL_WW_Random.append(Ave_L_[1])
                    
                    #Entropy of LONA for one and two wings setting

                    SE_W_LONA.append(Ave_E[0])
                    SE_WW_LONA.append(Ave_E[1])
                    SE_W_Random.append(Ave_E_[0])
                    SE_WW_Random.append(Ave_E_[1])
                    

                    
                E_W_LONA_.append(np.mean(E_W_LONA))
                E_WW_LONA_.append(np.mean(E_WW_LONA))
                E_W_Random_.append(np.mean(E_W_Random))
                E_WW_Random_.append(np.mean(E_WW_Random))
                
                L_W_LONA_.append(np.mean(L_W_LONA))
                L_WW_LONA_.append(np.mean(L_WW_LONA))
                L_W_Random_.append(np.mean(L_W_Random))
                L_WW_Random_.append(np.mean(L_WW_Random)) 

                SE_W_LONA_.append(np.mean(SE_W_LONA))
                SE_WW_LONA_.append(np.mean(SE_WW_LONA))
                SE_W_Random_.append(np.mean(SE_W_Random))
                SE_WW_Random_.append(np.mean(SE_WW_Random))
                
                SL_W_LONA_.append(np.mean(SL_W_LONA))
                SL_WW_LONA_.append(np.mean(SL_WW_LONA))
                SL_W_Random_.append(np.mean(SL_W_Random))
                SL_WW_Random_.append(np.mean(SL_WW_Random)) 


            data[str(self.d)]['E_W_LONA'] = E_W_LONA_
            data[str(self.d)]['E_WW_LONA'] = E_WW_LONA_                    
            data[str(self.d)]['E_W_Random'] = E_W_Random_                    
            data[str(self.d)]['E_WW_Random'] = E_WW_Random_                    

            data[str(self.d)]['L_W_LONA'] = L_W_LONA_
            data[str(self.d)]['L_WW_LONA'] = L_WW_LONA_                    
            data[str(self.d)]['L_W_Random'] = L_W_Random_                    
            data[str(self.d)]['L_WW_Random'] = L_WW_Random_                     
                    

            data[str(self.d)]['SE_W_LONA'] = SE_W_LONA_
            data[str(self.d)]['SE_WW_LONA'] = SE_WW_LONA_                    
            data[str(self.d)]['SE_W_Random'] = SE_W_Random_                    
            data[str(self.d)]['SE_WW_Random'] = SE_WW_Random_                    

            data[str(self.d)]['SL_W_LONA'] = SL_W_LONA_
            data[str(self.d)]['SL_WW_LONA'] = SL_WW_LONA_                    
            data[str(self.d)]['SL_W_Random'] = SL_W_Random_                    
            data[str(self.d)]['SL_WW_Random'] = SL_WW_Random_                            
  
        
        return data     
      
    def Raw_Data(self): 
        Data = 10000*np.ones((self.N,self.N))

        with open('Interpolated_NYM_250_DEC_2023_Omega.json') as json_file: 
        
            data0 = json.load(json_file) 
        number_of_data = len(data0)-1
        List = []
        i=0
        while(i<(self.N)):
            a = int(number_of_data*np.random.rand(1)[0]+1)
            if a > number_of_data:
                a==number_of_data
            if not a in List:
                List.append(a)
                i = i +1
                
        for i in range(len(List)):
            ID1 = List[i]
            for j in range(len(List)):
                if not i==j:
                    ID2 = List[j] 
                    I_key = data0[ID2]['i_key']
                    In_Latency = data0[ID1]['latency_measurements'][str(I_key)]
                    delay_distance = In_Latency
                    if delay_distance == 0:
                        delay_distance =1
                    Data[i,j] = abs(delay_distance)/2000
            
        return Data           

    
    


    def Random_A(self,Matrix):  
        data0 = {}
        #Adding client to chains laytency
        for j in range(self.U):
            data0['Client'+str(j+1)] = To_list(Matrix[j,j+1:j+1+self.d])
        
        #Addig inter_chains latency
        
        for k in range(self.W-1):
            for z in range(k*self.d,self.d*(k+1)):
                data0['Chain'+str(z+1)] = To_list(Matrix[z,self.d*(k+1):self.d*(k+2)])

        for i in range(self.d*self.W):
            data0['Within_Chain'+str(i+1)] = 0 
            
        data0['Mixnodes'] = Matrix
        data0['List'] = [i for i in range(self.d*self.W)]
        return data0      
    
    
    
    
    def Simulator_x(self,corrupted_Mix,Mix_Dict): 

        Mixes = [] #All mix nodes
        GateWays = {}
        env = simpy.Environment()    #simpy environment
        capacity=[]
        for j in range(self.N):# Generating capacities for mix nodes  
            c = simpy.Resource(env,capacity = self.CAP)
            capacity.append(c)           
        for i in range(self.N):#Generate enough instantiation of mix nodes  
            ll = i +1
            X = corrupted_Mix['PM%d' %ll]
            x = Mix(env,'M%02d' %i,capacity[i],X,self.Targets,self.delay1)
            Mixes.append(x)
        
 
        for i in range(self.U):#Generate enough instantiation of GateWays  
            ll = i +1

            gw = GateWay(env,'GW%02d' %i,0.00001)
            G = 'G' + str(ll)
            GateWays[G] = gw


       

        MNet = MixNet(env,Mixes,GateWays)  #Generate an instantiation of the mix net
        random.seed(42)  
        

        Process = Message_Genartion_and_mix_net_processing(env,Mixes,capacity,Mix_Dict,MNet,self.Targets,self.delay2,self.H_N,self.rate)

        env.process(Process.Prc())  #process the simulation

        env.run(until = self.run)  #Running time

        
        Latencies = MNet.LL
       
        Latencies_T = MNet.LT
        Distributions = np.matrix(MNet.EN)
        DT = np.transpose(Distributions)
        ENT = []

        for i in range(self.Targets):
            llll = DT[i,:].tolist()[0]
            ENT.append(Ent_x(llll))
        return Latencies, Latencies_T,ENT
    
    def EL_Sim(self,T_List,Iterations):
        Names = ['LAR','GPR','EXP','LAS']

        with open('Results/Basic_data_2.json','rb') as json_file:
            data0 = pickle.load(json_file)        
        corrupted_Mix = {}
        
        for i in range(self.N):
            corrupted_Mix['PM'+str(i+1)] = False
##########################Simulations#######################################################   
        Latency_LAS = []
        Latency_LAS_T = []    
        Entropy_LAS = []
        Latency_LAR = []
        Latency_LAR_T = []    
        Entropy_LAR = []
        Latency_GPR = []
        Latency_GPR_T = []    
        Entropy_GPR = []
        Latency_EXP = []
        Latency_EXP_T = []    
        Entropy_EXP = []

###########################################LAS####################################################################################   
        for j in range(len(T_List)):
            alpha = T_List[j]       
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            

            for i in range(Iterations):
                            
                List = data0['It'+str(i+1)]['LAS'][j]
                Mix_Dict = self.Basic_2_Sim(List)

                Latencies, Latencies_T,ENT = self.Simulator_x(corrupted_Mix,Mix_Dict)
                End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                Message_Entropy_Vector = Message_Entropy_Vector + ENT  
                    
            Latency_LAS.append(End_to_End_Latancy_Vector)
            Latency_LAS_T.append(End_to_End_Latancy_Vector_T)
            Entropy_LAS.append(Message_Entropy_Vector)        
        
        

###########################################LAR####################################################################################   
        for j in range(len(T_List)):
            alpha = T_List[j]       
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            

            for i in range(Iterations):
                            
                List = data0['It'+str(i+1)]['LAR'][j]
                Mix_Dict = self.Basic_2_Sim(List)

                Latencies, Latencies_T,ENT = self.Simulator_x(corrupted_Mix,Mix_Dict)
                End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                Message_Entropy_Vector = Message_Entropy_Vector + ENT  
                    
            Latency_LAR.append(End_to_End_Latancy_Vector)
            Latency_LAR_T.append(End_to_End_Latancy_Vector_T)
            Entropy_LAR.append(Message_Entropy_Vector)
        


###########################################GPR####################################################################################   
        for j in range(len(T_List)):
            alpha = T_List[j]       
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            

            for i in range(Iterations):
                            
                List = data0['It'+str(i+1)]['GPR'][j]
                Mix_Dict = self.Basic_2_Sim(List)

                Latencies, Latencies_T,ENT = self.Simulator_x(corrupted_Mix,Mix_Dict)
                End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                Message_Entropy_Vector = Message_Entropy_Vector + ENT  
                    
            Latency_GPR.append(End_to_End_Latancy_Vector)
            Latency_GPR_T.append(End_to_End_Latancy_Vector_T)
            Entropy_GPR.append(Message_Entropy_Vector)


###########################################EXP####################################################################################   
        for j in range(len(T_List)):
            alpha = T_List[j]       
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            

            for i in range(Iterations):

                            
                List = data0['It'+str(i+1)]['EXP'][j]
                Mix_Dict = self.Basic_2_Sim(List)

                Latencies, Latencies_T,ENT = self.Simulator_x(corrupted_Mix,Mix_Dict)
                End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                Message_Entropy_Vector = Message_Entropy_Vector + ENT  
                    
            Latency_EXP.append(End_to_End_Latancy_Vector)
            Latency_EXP_T.append(End_to_End_Latancy_Vector_T)
            Entropy_EXP.append(Message_Entropy_Vector)

###################################################################################            
#################################Saving the data###################################     
        df = {'Tau':T_List,
            'Latency_LAR' : Latency_LAR,
            'Entropy_LAR' : Entropy_LAR,     
            'Latency_GPR' : Latency_GPR,
            'Entropy_GPR' : Entropy_GPR, 
            'Latency_EXP' : Latency_EXP,
            'Entropy_EXP' : Entropy_EXP,
            'Latency_LAS' : Latency_LAS,
            'Entropy_LAS' : Entropy_LAS            
                              }

        dics = (df)
        with open( 'Results/' +'Sim_Basic_.json','w') as df_sim:
            json.dump(dics,df_sim)    
        
        
        




    def E2E(self,e2e,Iterations,T_List,item):

        
        with open('Results/Basic_data_2.json','rb') as json_file:
            data0 = pickle.load(json_file)  


        
#################################################################################         
        Latency  = []
        Latency_T = []    
        Entropy   = []
        E_A       = []
        L_A       = []

        corrupted_Mix = {}

        for k in range(self.N):
            corrupted_Mix['PM'+str(k+1)] = False


###########################################Uniform ##############################   
        for j in range(len(T_List)):

            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []  
            E_A_ = []
            L_A_ = []

            for i in range(Iterations):

                            
                Ave_Latency = LP_AVE(data0['It'+str(i+1)][item][j][0],data0['It'+str(i+1)][item][j][1])
                E_analytic = self.T_entropy(data0['It'+str(i+1)][item][j][1])
                self.d1 = (e2e - Ave_Latency)/3
                Mix_Dict = self.Basic_2_Sim(data0['It'+str(i+1)][item][j])


                E_A_.append(E_analytic)
                L_A_.append(Ave_Latency)
                Latencies, Latencies_T,ENT = self.Simulator_x(corrupted_Mix,Mix_Dict)
                End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                Message_Entropy_Vector = Message_Entropy_Vector + ENT  
                    
            Latency.append(End_to_End_Latancy_Vector)
            Latency_T.append(End_to_End_Latancy_Vector_T)
            Entropy.append(Message_Entropy_Vector)
            E_A.append(E_A_)
            L_A.append(L_A_)
        Sim_L_mean = Medd(Latency)
        Sim_E_mean = Medd(Entropy)
        Ana_L_mean = Medd(L_A)
        Ana_E_mean = Medd(E_A)


            

        df = {'Sim_E_mean':Sim_E_mean,
              'Sim_L_mean':Sim_L_mean,
              'Entropy_A':Ana_E_mean,
              'Latency_A1':Ana_L_mean}
        
        with open('Results/Data_Tradeoffs'+item+'.pkl','wb') as file:
            pickle.dump(df,file)

        


    def Random_x(self,N,C):
        Index = []
        counter = 0
        while counter <C:
            New_Mix = int((N)*np.random.rand(1)[0])
            if (not New_Mix in Index) and (not New_Mix==N):
                Index.append(New_Mix)
                counter +=1
        return Index
                

    def Greedy_x(self,Laytency_Matrix_of_mixnodes,Capacity):
        beta = [1]*len(Laytency_Matrix_of_mixnodes)
        L_M = np.copy(Laytency_Matrix_of_mixnodes)
        Max_Omega = Capacity
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
        
        

    def Greedy_For_Fairness_x(self, capacity, R_List_):
        """
        Distributes capacity fairly across L layers based on values in R_List_.
    
        Parameters:
            capacity (int): Total capacity to distribute. Should be between 0 and len(R_List_[0]) * 3.
            R_List_ (list of np.ndarray): List of 2D numpy arrays (matrices), each of size W x W.
    
        Returns:
            list of lists: Selected indices for each layer.
        """
        L = len(R_List_) + 1  # Number of layers
        W = len(R_List_[0])   # Width of each matrix
        Omega_L = capacity / L  # Fair share of capacity per layer
        Omega = int(W * L)      # Total number of nodes theoretically
        beta = np.ones((L, W))  # Availability/weighting matrix
        P_x = np.full(W, 1/W)   # Initial probability distribution (unused in final logic)
        
        CNodes = []  # To store selected nodes for each layer
    
        # First layer: random selection
        selected = set()
        while len(selected) < Omega_L:
            idx = np.random.randint(W)
            if idx not in selected:
                selected.add(idx)
        
        CNodes.append(list(selected))
    
        # Remaining layers: greedy selection based on previous layer's connectivity
        for l in range(L - 1):
            R_List = [np.copy(m) for m in R_List_]  # Deep copy to avoid mutation
            selected = []
            cap = 0
            
            while cap < Omega_L:
                if l >= len(R_List):
                    break
    
                prev_nodes = CNodes[l]
                scores = np.sum(R_List[l][prev_nodes], axis=0)
    
                # Find the index of the highest-scoring node not yet selected
                sorted_indices = np.argsort(-scores)  # Descending order
                for idx in sorted_indices:
                    if beta[l+1][idx] > 0:
                        selected.append(idx)
                        cap += beta[l+1][idx]
                        beta[l+1][idx] = -10000  # Mark as used
                        R_List[l][:, idx] = -10000  # Mask out for future
                        break  # Go back to while loop
    
            CNodes.append(selected)
    
        return CNodes
    def Corruption_Mix(self,List,N):
        Corrupted_Mix = {}       
        for i in range(N):
            Corrupted_Mix['PM'+str(i+1)] = False
            
        for item in List:
            Corrupted_Mix['PM'+str(int(item)+1)] = True  
        
        return Corrupted_Mix
        
    def FCP_x(self,R_List,P,List_C,W,TYPE = False):
        #Let TYPE be True if you applied greedy_fairness
        R1 = np.matrix(R_List[0])
        R2 = np.matrix(R_List[1]) 

        
        if not TYPE:
            List = []
            
            for i in range(self.W):
                
                List_ = []
                for item in List_C:
                    
                    if W*i <= item < W*(i+1):
                        List_.append(item-W*i)
                List.append(List_)
        else:
            List = List_C
            


        Path_C  = 0
        for i in (List[0]):
            for j in (List[1]):
                for k in (List[2]):
                    
                    Path_C += P[i]*R1[i,j]*R2[j,k]

        if Path_C>1:
            pass

        return Path_C


    def FCP_Analysis(self,Iterations,T_List):
        self.CF = int(self.N*0.15)

        with open('Results/Basic_data_2.json','rb') as json_file:
            data0 = pickle.load(json_file)          
        W,WWW = np.shape(data0['It1']['LAS'][0][1][0])
        Names = ['LAS','LAR','GPR','EXP']
         
        
        data_1 = {}
        data_sim = {}
        for typ in Names:
            data_sim0 = {}
            F_C_P_  = [] 
            F_G_P_   = []                           
            F_R_P_   = []  

            F_C_B_  = [] 
            F_G_B_   = []                           
            F_R_B_   = []  

            F_C_N_  = [] 
            F_G_N_   = []                           
            F_R_N_   = []   
                    
            for i in range(len(T_List)):
                data_sim1 = {}
                F_C_P = [] 
                F_G_P = []                           
                F_R_P = []  

                F_C_B = [] 
                F_G_B = []                           
                F_R_B = []  

                F_C_N = [] 
                F_G_N = []                           
                F_R_N = []                         
                for It in range(Iterations):
                    Matrix_xx = data0['It'+str(It+1)]['Mixnodes']


                    datum = data0['It'+str(It+1)][typ][i][1]
                    P     = To_list(np.sum(datum[0] ,axis=0)/W)
                    R1    = datum[1]
                    R2    = datum[2]
                    
                    List_c_Greedy = self.Greedy_For_Fairness_x(self.CF, [R1,R2])
                    List_c_Close  =  self.Greedy_x(Matrix_xx,self.CF)
                    List_c_Random =  self.Random_x(self.N,self.CF)
                    greedy_sim = []
                    for items in List_c_Greedy:
                        greedy_sim += items
                    
                    sim1 = self.Corruption_Mix(greedy_sim,self.N)
                    sim2 = self.Corruption_Mix(List_c_Close,self.N)
                    sim3 = self.Corruption_Mix(List_c_Random,self.N)
                    
                    data_sim1['It'+str(It+1)] = {'G': sim1, 'C': sim2 , 'R': sim3,'Routes':data0['It'+str(It+1)][typ][i]}                     

                    F_G_P.append(self.FCP_x([R1,R2],P,List_c_Greedy,W,True))
                        
                    F_C_P.append(self.FCP_x([R1,R2],P,List_c_Close,W,False))

                    F_R_P.append(self.FCP_x([R1,R2],P,List_c_Random,W,False))
                    
                    #Balanced case##############################################################
                    P_B     = To_list(np.sum(Balance_E(datum[0],self.DP) ,axis=0)/W)                    
                    B1 = Balance_E(R1,self.DP)
                    B2 = Balance_E(R2,self.DP)
                    
                    List_c_Greedy_B = self.Greedy_For_Fairness_x(self.CF, [B1,B2])

                    F_G_B.append(self.FCP_x([B1,B2],P_B,List_c_Greedy_B,W,True))
                        
                    F_C_B.append(self.FCP_x([B1,B2],P_B,List_c_Close,W,False))

                    F_R_B.append(self.FCP_x([B1,B2],P_B,List_c_Random,W,False))                    
                    

                    
                    #Noise Case#####################################################################
                    P_N     = To_list(np.sum(self.Noise(datum[0],0.02) ,axis=0)/W)                     
                    N1 = self.Noise(R1,0.02)
                    N2 = self.Noise(R2,0.02)  
                   
                    List_c_Greedy_N = self.Greedy_For_Fairness_x(self.CF, [N1,N2])

                    F_G_N.append(self.FCP_x([N1,N2],P_N,List_c_Greedy_N,W,True))
                        
                    F_C_N.append(self.FCP_x([N1,N2],P_N,List_c_Close,W,False))

                    F_R_N.append(self.FCP_x([N1,N2],P_N,List_c_Random,W,False))
                data_sim0['Tau'+str(i)] = data_sim1

                F_G_P_.append(Medd([F_G_P])[0])
                F_C_P_.append(Medd([F_C_P])[0])                        
                F_R_P_.append(Medd([F_R_P])[0]) 
                
                F_G_B_.append(Medd([F_G_B])[0])
                F_C_B_.append(Medd([F_C_B])[0])                        
                F_R_B_.append(Medd([F_R_B])[0])                 
                
                F_G_N_.append(Medd([F_G_N])[0])
                F_C_N_.append(Medd([F_C_N])[0])                        
                F_R_N_.append(Medd([F_R_N])[0])
                
                
            data_sim[typ] = data_sim0

            data_1[typ] = { 'Plian':{ 'G':F_G_P_,'C':F_C_P_,'R':F_R_P_},
                           'Balanced':{ 'G':F_G_B_,'C':F_C_B_,'R':F_R_B_},
                           'Noise':{ 'G':F_G_N_,'C':F_C_N_,'R':F_R_N_}
                           }



            
            
        with open('Results/FCP_EXP_.pkl','wb') as file:

            pickle.dump(data_1, file)          

        with open('Results/FCP_Sim_Initial.pkl','wb') as file:

            pickle.dump(data_sim, file)                  

    

    
    def FCP_Analysis_Bx(self,Iterations):
        CF_List = [0.1,0.13,0.16,0.19,0.22]

        with open('Results/Basic_data_2.json','rb') as json_file:
            data0 = pickle.load(json_file)          
        W,WWW = np.shape(data0['It1']['LAS'][0][1][0])
        Names = ['LAS','LAR','GPR','EXP']
         
        
        data_1 = {}
        for typ in Names:
            F_C_P_  = [] 
            F_G_P_   = []                           
            F_R_P_   = []  

            F_C_B_  = [] 
            F_G_B_   = []                           
            F_R_B_   = []  

            F_C_N_  = [] 
            F_G_N_   = []                           
            F_R_N_   = []   
                    
            for cff in (CF_List):
                self.CF = int(cff*self.N)
                i = 3
                data_sim1 = {}
                F_C_P = [] 
                F_G_P = []                           
                F_R_P = []  

                F_C_B = [] 
                F_G_B = []                           
                F_R_B = []  

                F_C_N = [] 
                F_G_N = []                           
                F_R_N = []                         
                for It in range(Iterations):
                    Matrix_xx = data0['It'+str(It+1)]['Mixnodes']


                    datum = data0['It'+str(It+1)][typ][i][1]
                    P     = To_list(np.sum(datum[0] ,axis=0)/W)
                    R1    = datum[1]
                    R2    = datum[2]
                    
                    List_c_Greedy = self.Greedy_For_Fairness_x(self.CF, [R1,R2])
                    List_c_Close  =  self.Greedy_x(Matrix_xx,self.CF)
                    List_c_Random =  self.Random_x(self.N,self.CF)

                   

                    F_G_P.append(self.FCP_x([R1,R2],P,List_c_Greedy,W,True))
                        
                    F_C_P.append(self.FCP_x([R1,R2],P,List_c_Close,W,False))

                    F_R_P.append(self.FCP_x([R1,R2],P,List_c_Random,W,False))
                    
                    #Balanced case##############################################################
                    P_B     = To_list(np.sum(Balance_E(datum[0],self.DP) ,axis=0)/W)                    
                    B1 = Balance_E(R1,self.DP)
                    B2 = Balance_E(R2,self.DP)
                    
                    List_c_Greedy_B = self.Greedy_For_Fairness_x(self.CF, [B1,B2])

                    F_G_B.append(self.FCP_x([B1,B2],P_B,List_c_Greedy_B,W,True))
                        
                    F_C_B.append(self.FCP_x([B1,B2],P_B,List_c_Close,W,False))

                    F_R_B.append(self.FCP_x([B1,B2],P_B,List_c_Random,W,False))                    
                    

                    
                    #Noise Case#####################################################################
                    P_N     = To_list(np.sum(self.Noise(datum[0],0.02) ,axis=0)/W)                     
                    N1 = self.Noise(R1,0.02)
                    N2 = self.Noise(R2,0.02)  
                   
                    List_c_Greedy_N = self.Greedy_For_Fairness_x(self.CF, [N1,N2])

                    F_G_N.append(self.FCP_x([N1,N2],P_N,List_c_Greedy_N,W,True))
                        
                    F_C_N.append(self.FCP_x([N1,N2],P_N,List_c_Close,W,False))

                    F_R_N.append(self.FCP_x([N1,N2],P_N,List_c_Random,W,False))

                F_G_P_.append(Medd([F_G_P])[0])
                F_C_P_.append(Medd([F_C_P])[0])                        
                F_R_P_.append(Medd([F_R_P])[0]) 
                
                F_G_B_.append(Medd([F_G_B])[0])
                F_C_B_.append(Medd([F_C_B])[0])                        
                F_R_B_.append(Medd([F_R_B])[0])                 
                
                F_G_N_.append(Medd([F_G_N])[0])
                F_C_N_.append(Medd([F_C_N])[0])                        
                F_R_N_.append(Medd([F_R_N])[0])
                


            data_1[typ] = { 'Plian':{ 'G':F_G_P_,'C':F_C_P_,'R':F_R_P_},
                           'Balanced':{ 'G':F_G_B_,'C':F_C_B_,'R':F_R_B_},
                           'Noise':{ 'G':F_G_N_,'C':F_C_N_,'R':F_R_N_}
                           }



            
            
        with open('Results/FCP_EXP_B.pkl','wb') as file:

            pickle.dump(data_1, file)   
            
            
            
            
            
            

    def FCP_Sim_x(self,T_List,Iterations,name):
        Names = ['LAR','GPR','EXP','LAS']

        with open('Results/FCP_Sim_Initial.pkl','rb') as file:
            data1 = pickle.load( file)             
##########################Simulations#######################################################         
        Latency_R = []
        Latency_R_T = []    
        Entropy_R = []
        Latency_G = []
        Latency_G_T = []    
        Entropy_G = []
        Latency_C = []
        Latency_C_T = []    
        Entropy_C = []


###########################################Random####################################################################################   
        for j in range(len(T_List)):
            alpha = T_List[j]       
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            

            for i in range(Iterations):
                corrupted_Mix = data1[name]['Tau'+str(j)]['It'+str(i+1)]['R']
                            
                List = data1[name]['Tau'+str(j)]['It'+str(i+1)]['Routes']
                Mix_Dict = self.Basic_2_Sim(List)

                Latencies, Latencies_T,ENT = self.Simulator_x(corrupted_Mix,Mix_Dict)
                End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                Message_Entropy_Vector = Message_Entropy_Vector + ENT  
                    
            Latency_R.append(End_to_End_Latancy_Vector)
            Latency_R_T.append(End_to_End_Latancy_Vector_T)
            Entropy_R.append(Message_Entropy_Vector)
        


###########################################Close####################################################################################   
        for j in range(len(T_List)):
            alpha = T_List[j]       
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            

            for i in range(Iterations):
                corrupted_Mix = data1[name]['Tau'+str(j)]['It'+str(i+1)]['C']
                            
                List = data1[name]['Tau'+str(j)]['It'+str(i+1)]['Routes']
                Mix_Dict = self.Basic_2_Sim(List)

                Latencies, Latencies_T,ENT = self.Simulator_x(corrupted_Mix,Mix_Dict)
                End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                Message_Entropy_Vector = Message_Entropy_Vector + ENT  
                    
            Latency_C.append(End_to_End_Latancy_Vector)
            Latency_C_T.append(End_to_End_Latancy_Vector_T)
            Entropy_C.append(Message_Entropy_Vector)

###################################################################################   
            
###########################################Greedy####################################################################################   
        for j in range(len(T_List)):
            alpha = T_List[j]       
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            

            for i in range(Iterations):
                corrupted_Mix = data1[name]['Tau'+str(j)]['It'+str(i+1)]['G']
                            
                List = data1[name]['Tau'+str(j)]['It'+str(i+1)]['Routes']
                Mix_Dict = self.Basic_2_Sim(List)

                Latencies, Latencies_T,ENT = self.Simulator_x(corrupted_Mix,Mix_Dict)
                End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                Message_Entropy_Vector = Message_Entropy_Vector + ENT  
                    
            Latency_G.append(End_to_End_Latancy_Vector)
            Latency_G_T.append(End_to_End_Latancy_Vector_T)
            Entropy_G.append(Message_Entropy_Vector)

################################################################################### 
#################################Saving the data###################################     
        df = {'Tau':T_List,
            'Entropy_R' : Entropy_R,     
            'Entropy_G' : Entropy_G, 
            'Entropy_C' : Entropy_C
                              }

        dics = (df)
        with open( 'Results/'+name +'Sim_FCP.json','w') as df_sim:
            json.dump(dics,df_sim)    
        
        
    


    def Noise_EXPP_x(self,T_List,Iterations,i):
        W = self.d
        self.CF = int(0.2*self.N)
        Names = ['LAR','LAS','GPR','EXP']

        with open('Results/Basic_data_2.json','rb') as json_file:
            data0 = pickle.load(json_file)

        data = {}

        for name in Names:
            data[name] = {}
        for item in Names:
            E_ = []
            F_ = []           

            for noise_level in T_List:
                E = []
                F = []


                for It in range(0,Iterations):
                    Matrix_xx = data0['It'+str(It+1)]['Mixnodes']
                    #Adding noise to the rouitng distributions for LONA
                    N1 = self.Noise(data0['It'+str(It+1)][item][i][1][0],noise_level)
                    N2 = self.Noise(data0['It'+str(It+1)][item][i][1][1],noise_level)  
                    N3 = self.Noise(data0['It'+str(It+1)][item][i][1][2],noise_level)
                    P     = To_list(np.sum(N1 ,axis=0)/W)
                    List_c_Greedy = self.Greedy_For_Fairness_x(self.CF, [N2,N3])

                    F.append(self.FCP_x([N2,N3],P,List_c_Greedy,W,True))


                    E.append(self.T_entropy([N1,N2,N3]))


                
                E_.append(np.mean(E))
                F_.append(np.mean(F))
                
            data[item]['E']     = E_                    
            data[item]['F']     = F_                  


        with open('Results/CRG_Noise.pkl','wb') as json_file:
            pickle.dump(data,json_file)
  
    





    def data_FCP(self,Iteration):
        data0 = {}
        W = {'NYM':80,'RIPE':200}
        
        for Data in ['NYM','RIPE']:
            data1 = {}
            for It in range(Iteration):

                Matrix = self.Data_Set_General[Data]['It'+str(It+1)]['DNA']['Matrix']
                Latency_List = self.Data_Set_General[Data]['It'+str(It+1)]['DNA']['Latency_List']
                Positions = self.Data_Set_General[Data]['It'+str(It+1)]['DNA']['Positions']
                Loc = self.Data_Set_General[Data]['It'+str(It+1)]['DNA']['Loc']
                O_ = self.Data_Set_General[Data]['It'+str(It+1)]['DNA']['Omega']
                Omega = self.Data_Set_General[Data]['It'+str(It+1)]['DNA']['x']
                _ = self.Data_Set_General[Data]['It'+str(It+1)]['DNA']['xx']

                O_1 = []
                for item in _:
                    O_1 += item

                
                data4 = {'Latency_List': Latency_List,'Omega':O_, 'Positions':Positions,'Loc':Loc,'beta':[Omega,O_1],'L_M':Matrix}

                data1['It'+str(It+1)] = data4
                
            data0[Data] = data1
                
        return data0  















    def LAMP_SC(self,Iterations,Data_type):

        self.L = 3
        self.WW = {'NYM':80 , 'RIPE':200}
        self.W1 = self.WW['NYM']
        self.W2 = self.WW['RIPE']
        self.CF = 0.3
        
        self.Data_Set_General = {'NYM':{},'RIPE':{}}
        self.Iterations = Iterations
        
        if True:
            with open('data0.pkl','rb') as pkl_file:
                data0 = pickle.load(pkl_file)

            for item in Data_type:
                
                for It in range(self.Iterations):
                    self.Data_Set_General[item]['It'+str(It+1)] = data0[item]['It'+str(It+1)]

        data_W1 = {}
        for i1 in range(self.W1*(self.L)):
            data_W1['PM'+str(i1+1)] = False
        data_W2 = {}
        for i1 in range(self.W2*(self.L)):
            data_W2['PM'+str(i1+1)] = False   
        self.Corrupted_Mix = {80:data_W1,200:data_W2}




        data = self.Data_Set_General        
        data_0 = self.data_FCP(Iterations)
        tau = 0.4
        r = 0.015
        data1 = {}

        for typ in Data_type:
            elements = [i for i in range(self.WW[typ])]

            data2 = {}
            Class_R = Routing((self.WW[typ]*3),3)
            L_0 = []
            H_0 = []
            W_0 = []
            HM_0 = []
            FCP_0 = []
            for It in range(self.Iterations):

                L_Mix = data[typ]['It'+str(It+1)]['DNA']['Latency_List'] 
                O_Mix = data[typ]['It'+str(It+1)]['DNA']['Omega'] 
                Latency_SC_Matrix = data[typ]['It'+str(It+1)]['DNA']['Matrix'] 
                SC_P = data[typ]['It'+str(It+1)]['DNA']['Positions'] 
                L_SC_Matrix = SC_Latency(Latency_SC_Matrix,SC_P,self.L)
                Helper = self.filter_SC(L_SC_Matrix,r,self.L)
                
                Routing1 = []
                Routing2 = {}
                for i_ in range(self.WW[typ]):
                    Routing2[str(i_)] = []

                P = [1/self.WW[typ]]*self.WW[typ]
                
                for I in range(len(Helper)):
                    L_temp = remove_elements_by_index(L_Mix[0][I],Helper[I][0])
                    ele_temp = remove_elements_by_index(elements.copy(),Helper[I][0])

                    r_temp = self.LARMIX(L_temp,tau)

                    r_policy = add_elements_by_index(r_temp,Helper[I][0]) 
                    Routing1.append(r_policy)
                    x_list = []
                    for i in ele_temp:
                            
                        L_temp1 = remove_elements_by_index(L_Mix[1][i],Helper[I][1])
                        r_temp1 = self.LARMIX(L_temp1,tau)                            
                        r_policy1 = add_elements_by_index(r_temp1,Helper[I][1]) 
                        Routing2[str(i)].append(r_policy1)

                R_Final = []
                for i_ in range(self.WW[typ]):
                    y = To_list(np.mean(np.matrix(Routing2[str(i_)]),axis=0))
                    
                    if len(y)==0:
                        y = [0]*self.WW[typ]
                        y[0] = 1
                    
                    R_Final.append(y)

                    
                        
                
                R1 = [np.matrix(Routing1),np.matrix(R_Final)]
                
                L1= L_Mix.copy()
                R11 = [To_list(R1[i]) for i in range(self.L-1)]
                Latency_Sim0,Entropy_Sim0 = self.Simp(L1,R11,P,config.n_scale,self.WW[typ],self.Corrupted_Mix[self.WW[typ]])
                HM_0 = HM_0 + Entropy_Sim0
                L11 = [np.matrix(L1[i]) for i in range(self.L-1)]
                L_0.append(Class_R.Latency_Measure(L11, R1, P))
                H_0.append(Class_R.Entropy_AVE(Class_R.Entropy_Transformation(R1),P))
 
                W_0.append(Class_R.Bandwidth_(R1, O_Mix, P))

                O_Mix_1 = np.matrix(data_0[typ]['It'+str(It+1)]['Omega'] )
                O_Mix1 = To_list(O_Mix_1)
                List_C = Greedy_For_Fairness(self.CF*self.L*self.WW[typ],O_Mix1,R1,self.L)
                FCP_0.append(self.FCP(R1,P,List_C,self.WW[typ],True))
              
            data2['L'] = np.mean(L_0)
            data2['H'] = np.mean(H_0)
            data2['W'] = np.mean(W_0)
            data2['HM'] = np.mean(HM_0)
            data2['FCP'] = np.mean(FCP_0)
            data1[typ] = data2
                   
                    
        return data1



    def Simp(self,List_L,List_R,P,nn,W,Corrupted_Mix):
        
        Mix_dict = {'Routing':List_R,'Latency':List_L,'First':P}
        
        
        
        
        Sim_ = Simulation_P(self.Targets,self.run,self.delay1,self.delay2,W*self.L,self.L )
        
        Latency_Sim,Entropy_Sim = Sim_.Simulator(Corrupted_Mix,Mix_dict,nn)
        
        
        return Latency_Sim, Entropy_Sim      



    def LARMIX(self,LIST_,Tau):#We materealize our function for making the trade off
        #In this function just for one sorted distribution

        t = Tau
        A, mapping = self.sort_and_get_mapping(LIST_)
        T = 1-t

        B=[]
        D=[]
    
    
        r = 1
        for i in range(len(A)):
            j = i
            J = (j*(1/(t**(r))))**(1-t)
    
            E = math.exp(-1)
            R = E**J
    
            B.append(R)
            A[i] = A[i]**(-T)
    
            g = A[i]*B[i]
    
            D.append(g)
        n=sum(D)
        for l in range(len(D)):
            D[l]=D[l]/n
        restored_list = self.restore_original_list(D, mapping)
        return restored_list



    def sort_and_get_mapping(self,initial_list):
        # Sort the initial list in ascending order and get the sorted indices
        sorted_indices = sorted(range(len(initial_list)), key=lambda x: initial_list[x])
        sorted_list = [initial_list[i] for i in sorted_indices]
    
        # Create a mapping from sorted index to original index
        mapping = {sorted_index: original_index for original_index, sorted_index in enumerate(sorted_indices)}
    
        return sorted_list, mapping
    
    def restore_original_list(self,sorted_list, mapping):
        # Create the original list by mapping each element back to its original position
        original_list = [sorted_list[mapping[i]] for i in range(len(sorted_list))]
        return original_list        





    def filter_SC(self,matrix, threshold,L):
        W = int(len(matrix)/L)
        flist = []
        for i in range(W):
            
            a = self.filter_matrix_entries(To_list(matrix[i,W:2*W]),threshold)
            b = self.filter_matrix_entries(To_list(matrix[i,2*W:3*W]),threshold)

            flist.append([a,b])

                
        return flist





    def FCP(self,R_List,P,List_C,W,TYPE = False):
        R1 = np.matrix(R_List[0])
        R2 = np.matrix(R_List[1]) 
        
        if not TYPE:
            List = []
            
            for i in range(3):
                
                List_ = []
                for item in List_C:
                    
                    if W*i <= item < W*(i+1):
                        List_.append(item-W*i)
                List.append(List_)
        else:
            List = List_C
            


        Path_C  = 0
        for i in (List[0]):
            for j in (List[1]):
                for k in (List[2]):
                    
                    Path_C += P[i]*R1[i,j]*R2[j,k]
        return Path_C

    def filter_matrix_entries(self,matrix, threshold):
        flist = []
        for i in range(len(matrix)):
            
            if matrix[i] > threshold:
                
                flist.append(i)

        if len(flist)==len(matrix):
            List = matrix.copy()
            length = round(0.02*len(matrix))
            Indices = []
            for j in range(length):
                
                Index = List.index(min(List))
                Indices.append(Index)
                List[Index] = 100000000000
            out = remove_elements_by_index(flist,Indices)
            
            return out
                
                
        return flist





    def LAMP_MC(self,Iterations,Data_type):

        self.L = 3
        self.WW = {'NYM':80 , 'RIPE':200}
        self.W1 = self.WW['NYM']
        self.W2 = self.WW['RIPE']
        self.CF = 0.3
        
        self.Data_Set_General = {'NYM':{},'RIPE':{}}
        self.Iterations = Iterations
        
        if True:
            with open('data0.pkl','rb') as pkl_file:
                data0 = pickle.load(pkl_file)

            for item in Data_type:
                
                for It in range(self.Iterations):

                    self.Data_Set_General[item]['It'+str(It+1)] = data0[item]['It'+str(It+1)]



        data_W1 = {}
        for i1 in range(self.W1*(self.L)):
            data_W1['PM'+str(i1+1)] = False
        data_W2 = {}
        for i1 in range(self.W2*(self.L)):
            data_W2['PM'+str(i1+1)] = False   
        self.Corrupted_Mix = {80:data_W1,200:data_W2}
        
        
        
        
        
        data = self.Data_Set_General  
        data_0 = self.data_FCP(self.Iterations)
        tau = 0.6
        r = 0.015
        data1 = {}
        self.Data_type = Data_type

        for typ in self.Data_type:

            data2 = {}
            Class_R = Routing((self.WW[typ]*self.L),self.L)
            L_0 = []
            H_0 = []
            W_0 = []
            HM_0 = []
            FCP_0 = []
            for It in range(self.Iterations):

                L_Mix = data[typ]['It'+str(It+1)]['DNA']['Latency_List'] 
                O_Mix = data[typ]['It'+str(It+1)]['DNA']['Omega'] 
                
                R1 = []
                P = [1/self.WW[typ]]*self.WW[typ]

                for Layer_num in range(self.L-1):
                    R_Mix1 = []
                    for W_num in range(self.WW[typ]):
                        indices = self.filter_matrix_entries(L_Mix[Layer_num][W_num],r)
                        
                        L_temp = remove_elements_by_index(L_Mix[Layer_num][W_num],indices)
                        
                        r_temp = self.LARMIX(L_temp,tau)
                        
                        r_policy = add_elements_by_index(r_temp,indices)

                        R_Mix1.append(r_policy)
                        
                    R1.append(np.matrix(R_Mix1))
                L1= L_Mix.copy()
                R11 = [To_list(R1[i]) for i in range(self.L-1)]

                Latency_Sim0,Entropy_Sim0 = self.Simp(L1,R11,P,config.n_scale,self.WW[typ],self.Corrupted_Mix[self.WW[typ]])
                HM_0 = HM_0 + Entropy_Sim0
                L11 = [np.matrix(L1[i]) for i in range(self.L-1)]
                L_0.append(Class_R.Latency_Measure(L11, R1, P))
                H_0.append(Class_R.Entropy_AVE(Class_R.Entropy_Transformation(R1),P))

                W_0.append(Class_R.Bandwidth_(R1, O_Mix, P))

                O_Mix_1 = np.matrix(data_0[typ]['It'+str(It+1)]['Omega'] )
                O_Mix1 = To_list(O_Mix_1)
                List_C = Greedy_For_Fairness(self.CF*self.L*self.WW[typ],O_Mix1,R1,self.L)
                FCP_0.append(self.FCP(R1,P,List_C,self.WW[typ],True))
              
            data2['L'] = np.mean(L_0)
            data2['H'] = np.mean(H_0)
            data2['W'] = np.mean(W_0)
            data2['HM'] = np.mean(HM_0)
            data2['FCP'] = np.mean(FCP_0)
            data1[typ] = data2
                   
                    
        return data1



    def MAP_Latency_Omega(self,Map,Matrix,Omega,L):
        
        N = len(Matrix)
        W = int(N/L)
        
        Latency_List = []
        Omega_List = []
        for i in range(L-1):
            List0 = []
            
            for j in range(W):
                List1 = []
                
                for k in range(W):

                    List1.append(Matrix[Map[i*W+j],Map[(i+1)*W+k]])
                List0.append(List1)
            Latency_List.append(List0)
                    
        for i in range(L):
            List0 = []
            
            for j in range(W):
                List0.append(Omega[i*W+j])
            Omega_List.append(List0)
                
                
        return Latency_List,Omega_List




    def EXP_LARMIX(self,Iterations,Data_type):

        self.L = 3
        self.WW = {'NYM':80 , 'RIPE':200}
        self.W1 = self.WW['NYM']
        self.W2 = self.WW['RIPE']
        self.CF = 0.3
        
        self.Data_Set_General = {'NYM':{},'RIPE':{}}
        self.Iterations = Iterations
        
        if True:
            with open('data0.pkl','rb') as pkl_file:
                data0 = pickle.load(pkl_file)

            for item in Data_type:
                
                for It in range(self.Iterations):

                    self.Data_Set_General[item]['It'+str(It+1)] = data0[item]['It'+str(It+1)]


        data_W1 = {}
        for i1 in range(self.W1*(self.L)):
            data_W1['PM'+str(i1+1)] = False
        data_W2 = {}
        for i1 in range(self.W2*(self.L)):
            data_W2['PM'+str(i1+1)] = False   
        self.Corrupted_Mix = {80:data_W1,200:data_W2}        
        data = self.Data_Set_General        
        data_0 = self.data_FCP(self.Iterations)
        tau = 0.8
        data1 = {}
        self.Data_type = Data_type

        for typ in self.Data_type:
            data2 = {}
            Class_R = Routing((self.WW[typ]*self.L),self.L)
            L_0 = []
            H_0 = []
            W_0 = []
            HM_0 = []
            FCP_0 = []
            for It in range(self.Iterations):
                Loc = data[typ]['It'+str(It+1)]['DNA']['Loc']
                A_Loc = np.matrix([np.random.rand(3) for itr in range(self.WW[typ]*self.L)])
                Loc += A_Loc
                Matrix_Mix = data[typ]['It'+str(It+1)]['DNA']['Matrix']
                Omega_Mix = data[typ]['It'+str(It+1)]['DNA']['x'] 

                    
                ####LARMIx preparations########################################################
                Class_cluster = Clustering(np.copy(Loc),'kmedoids',5,self.L,0)
                New_Loc = Class_cluster.Mixes
                Labels = Class_cluster.Labels
                
                Map = Class_cluster.Map
                Class_Div = Mix_Arrangements( np.copy(New_Loc),0, Labels,Class_cluster.Centers,0,1,False)
                Final_Loc_ = To_list(Class_Div.Topology)
                Final_Loc = []
                for item in Final_Loc_:
                    Final_Loc += item
                MAP_ = find_row_permutation(New_Loc,np.matrix(Final_Loc))
                MAP_Final = MAP_to_MAP(Map,MAP_)
                Latency_List_LARMIX, Omega_List_LARMIX = self.MAP_Latency_Omega(MAP_Final,Matrix_Mix,Omega_Mix,self.L)

                
                L_Mix = Latency_List_LARMIX 
                O_Mix66 = Omega_List_LARMIX
                O_Mix = []
                for item in O_Mix66:
                    O_Mix.append(Norm_List(item,self.WW[typ]))
                                    
                R22 = []
                P = [1/self.WW[typ]]*self.WW[typ]

                for Layer_num in range(self.L-1):
                    R_Mix1 = []
                    for W_num in range(self.WW[typ]):
                        r_temp = self.LARMIX(L_Mix[Layer_num][W_num],tau)
                        R_Mix1.append(r_temp)
                    R22.append(np.matrix(R_Mix1))
                    
                R1 = []
                
                for item_ in R22:
                    Class_greedy = Balanced_Layers(5,'IDK',self.WW[typ])
                    Class_greedy.IMD = np.copy(item_)
                    Class_greedy.Iterations()                
                    R1.append(Class_greedy.IMD)
                    
                    
                    
                L1= L_Mix.copy()

                R11 = [To_list(R1[i]) for i in range(self.L-1)]

                Latency_Sim0,Entropy_Sim0 = self.Simp(L1,R11,P,config.n_scale,self.WW[typ],self.Corrupted_Mix[self.WW[typ]])
                HM_0 = HM_0 + Entropy_Sim0
                L11 = [np.matrix(L1[i]) for i in range(self.L-1)]
                L_0.append(Class_R.Latency_Measure(L11, R1, P))
                H_0.append(Class_R.Entropy_AVE(Class_R.Entropy_Transformation(R1),P))
 
                W_0.append(Class_R.Bandwidth_(R1, O_Mix, P))

                O_Mix_1 = np.matrix(O_Mix)
                O_Mix1 = To_list(O_Mix_1)
                List_C = Greedy_For_Fairness(self.CF*self.L*self.WW[typ],O_Mix1,R1,self.L)
                FCP_0.append(self.FCP(R1,P,List_C,self.WW[typ],True))
              
            data2['L'] = np.mean(L_0)
            data2['H'] = np.mean(H_0)
            data2['W'] = np.mean(W_0)
            data2['HM'] = np.mean(HM_0)
            data2['FCP'] = np.mean(FCP_0)
            data1[typ] = data2
                   
                    
        return data1























