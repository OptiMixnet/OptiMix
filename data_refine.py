# -*- coding: utf-8 -*-
"""
This .py file helps adapt some of the data interfaces used in OptiMix.
"""
import numpy as np
import random

def get_random_element(lst):
    """
    Returns a random element from the given list.

    Args:
        lst (list): The input list.

    Returns:
        element: A randomly chosen element from the list.
    """
    if not lst:
        raise ValueError("The list is empty.")
    
    return random.choice(lst)

def To_list(List):

    List_ = List.tolist()
    if len(List_)==1:
        output = List_[0]
    else:
        output = List_
    
    return output

def Sum_list(Lists):
    List0 = [[]]


    
def first_b_minimum_with_indices(lst, b):
   """
   Returns the first b smallest elements and their indices in the original list.
   """
   if b > len(lst):
       raise ValueError("b cannot be greater than the length of the list")
   
   # Pair elements with their indices
   indexed_lst = list(enumerate(lst))
   # Sort by values
   sorted_by_value = sorted(indexed_lst, key=lambda x: x[1])
   # Get the first b elements
   result = sorted_by_value[:b]
   # Separate values and indices
   values = [x[1] for x in result]
   indices = [x[0] for x in result]
   return values, indices


def Cap_Min_dist(Omega,W_g,C_,b):
    s = len(Omega) - len([Omega[x]  for x in range(len(Omega)) if abs(Omega[x])>10000])
    if s < b:
        b = s
    List = [abs(-W_g + C_ + Omega[i]) for i  in range(len(Omega))]
    return first_b_minimum_with_indices(List, b)

def first_b_maximum_with_indices(lst, b):
    """
    Returns the first b largest elements and their indices in the original list.
    """
    if b > len(lst):
        raise ValueError("b cannot be greater than the length of the list")
    
    # Pair elements with their indices
    indexed_lst = list(enumerate(lst))
    # Sort by values in descending order
    sorted_by_value = sorted(indexed_lst, key=lambda x: x[1], reverse=True)
    # Get the first b elements
    result = sorted_by_value[:b]
    # Separate values and indices
    values = [x[1] for x in result]
    indices = [x[0] for x in result]
    return values, indices
    





def element_wise_sum(list_of_lists):
    """
    Computes the element-wise sum of a list of lists.
    Assumes all sublists have the same length.
    
    Args:
        list_of_lists (list of lists): Input list of lists with numbers.

    Returns:
        list: A list containing the element-wise sums.
    """
    # Check if the input list is not empty
    if not list_of_lists:
        return []
    
    # Initialize a result list with zeros of the same size as the first sublist
    result = [0] * len(list_of_lists[0])
    
    # Iterate through each sublist
    for sublist in list_of_lists:
        # Add each element of the sublist to the corresponding position in the result
        result = [x + y for x, y in zip(result, sublist)]
    
    return result
def random_order_list(L):
    # Create a list from 0 to L-1
    original_list = list(range(L))
    # Shuffle the list randomly
    random.shuffle(original_list)
    return original_list

class DNA(object):
    
    def __init__(self,Omega,Matrix,Layers,b):

        
        self.L = Layers
        self.N = Matrix.shape[0]
        self.W = int(self.N/self.L)
        self.Omega = Omega
        self.Mixes = Matrix
        self.b = b
        
    
    
    def DNA_Arrangement(self):
        Leader_Mix = []
        Test2 = np.copy(self.Mixes)        

        OM = np.copy(self.Omega)
        for i in range(self.L):
            x= round((self.N)*np.random.rand(1)[0])-1
            if x==-1:
                x=0
            while x in Leader_Mix:
                x= round((self.N)*np.random.rand(1)[0])-1
                if x==-1:
                    x=0
            Leader_Mix.append(x)
            
        Layers = {}
        for _ in range(len(Leader_Mix)):
            Layers['Layer'+str(_+1)] = []       
        for i in range(len(Leader_Mix)):
            Layers['Layer'+str(i+1)].append(Leader_Mix[i])
            Test2[:,Leader_Mix[i]] = 100000
            OM[Leader_Mix[i]] = -100000
            
            
            
        for j in range(self.W-1):
            Pro = random_order_list(self.L)

            for item in Pro:
                
                z0 = Layers['Layer'+str(item+1)]
                
                List_y = [To_list(Test2[k,:]) for k in z0]
                y0 = element_wise_sum(List_y)
                List0 , Index0 = first_b_minimum_with_indices(y0,self.b)
                Max_OM = -100
                for term in Index0:
                    if OM[term] > Max_OM:
                        Max_OM = OM[term]
                        Index = term
                Layers['Layer'+str(item+1)].append(Index)
                Test2[:,Index] = 10000
                OM[Index] = -10000
        
        OMEGA = []
        MIXNET = []

        for jj in range(self.L):
            Temp = []
            Ex = []
            for ii in range(self.W):
                Ex.append(Layers['Layer'+str(jj+1)][ii])
                Temp.append(self.Omega[Layers['Layer'+str(jj+1)][ii]])
            OMEGA.append(Temp)
            MIXNET.append(Ex)
                              
            
            
        return MIXNET,OMEGA

                
    def DNA_Arrangement_W(self):
        Leader_Mix = []
        Leader_Cap = []
        Test2 = np.copy(self.Mixes)        
 
        OM = np.copy(self.Omega)
        W_g = np.sum(OM)/self.L
        for i in range(self.L):
            x= round((self.N)*np.random.rand(1)[0])-1
            if x==-1:
                x=0
            while x in Leader_Mix:
                x= round((self.N)*np.random.rand(1)[0])-1
                if x==-1:
                    x=0
            Leader_Mix.append(x)
            Leader_Cap.append(OM[x])
            
        Layers = {}
        C_Layers = {}
        for _ in range(len(Leader_Mix)):
            Layers['Layer'+str(_+1)] = []       
            C_Layers['Layer'+str(_+1)] = Leader_Cap[_]
        for i in range(len(Leader_Mix)):
            Layers['Layer'+str(i+1)].append(Leader_Mix[i])
            Test2[:,Leader_Mix[i]] = 10000000
            OM[Leader_Mix[i]] = -10000000
            
            
            
        for j in range(self.W-1):
            Pro = random_order_list(self.L)

            for item in Pro:
                List0 , Index0 = Cap_Min_dist(OM, W_g, C_Layers['Layer'+str(item+1)], self.b)

                
                Index = get_random_element(Index0)

                Layers['Layer'+str(item+1)].append(Index)
                C_Layers['Layer'+str(item+1)] = C_Layers['Layer'+str(item+1)] + OM[Index]
                Test2[:,Index] = 1000000
                OM[Index] = -1000000

        
        OMEGA = []
        MIXNET = []

        for jj in range(self.L):
            Temp = []
            Ex = []
            for ii in range(self.W):
                Ex.append(Layers['Layer'+str(jj+1)][ii])
                Temp.append(self.Omega[Layers['Layer'+str(jj+1)][ii]])
            OMEGA.append(Temp)
            MIXNET.append(Ex)
                              
            
            
        return MIXNET,OMEGA

                
                        
                        
                        
    def Map(self,List):
        
        Test = np.zeros((self.N,self.N))
        for i in range(len(List)):
            Test[List[i],i] = 1
                      
        Test_= np.copy(Test)    
        Test__ = np.transpose(Test)
        Mix_net = (Test.dot(self.Mixes)).dot(Test__)   
        Positions = (Test_.dot(self.Loaction))       

        
        return Mix_net,Positions


    





    

