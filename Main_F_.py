# -*- coding: utf-8 -*-
"""
Main Functions:
Main_F.py contains a comprehensive set of functions for analyzing OptiMix on cascade (also butterfly) topologies, including all necessary functions to compute anonymity and latency metrics.
"""





import numpy as np
import json
import pickle
import random
import math
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

from CLUSTER    import Clustering


from MixNetArrangment      import Mix_Arrangements
from Greedy_LARMIX         import Balanced_Layers
from LARMix_Greedy         import Balanced_Layers

from Message_Genartion_and_mix_net_processing_ import Message_Genartion_and_mix_net_processing
from FCP_Functions_                            import Greedy_, Greedy, Random, Greedy_For_Fairness




























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
        



class Carmix_(object):
    
    def __init__(self,d,h,W,Targets,run,delay1,delay2,Mix_Threshold,Corrupted_Mix):
        self.d = d
        self.W = W
        self.h = h
        self.c_f = 0.4
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
        
    
    
    def Raw_Data(self): 
        Data = 10000*np.ones((self.N+self.U,self.N+self.U))

        with open('Nym_data.json') as json_file: 
        
            data0 = json.load(json_file) 
        number_of_data = len(data0)-1
        List = []
        i=0
        while(i<(self.N+self.U)):
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
    
    
    def LONA(self,Matrix):
        Backup = np.copy(Matrix)
        Mix_matrix = Matrix[:self.N,:self.N]
        Backup_1 = np.copy(Matrix)      
        data0 = {}
        Leader = random.sample(range(self.N), self.W*self.d)
        
        for item in Leader:
            Mix_matrix[:,item] = 100000
        List = [[Leader[i]] for i in range(len(Leader))]
        
        for i in range(self.h-1):
        
            S = int(len(Leader)*np.random.rand(1)[0])-1
            if S==-1:
                S=0
            for j in range(S,len(Leader)):
                Temp_List = To_list(Mix_matrix[List[j][i],:])
                M_next = Temp_List.index(min(Temp_List))
                List[j].append(M_next)
                Mix_matrix[:,M_next] = 100000
            if not S ==0:
                
                for j in range(0,S):
                    Temp_List = To_list(Mix_matrix[List[j][i],:])
                    M_next = Temp_List.index(min(Temp_List))
                    List[j].append(M_next)
                    Mix_matrix[:,M_next] = 100000
        LIST = List.copy()            
        
        #Adding client to chains laytency
        for j in range(self.N,self.N+self.U):
            Temp_set = []
            for k in range(self.d):
                Temp_set.append(Backup[Leader[k],j])
            data0['Client'+str(j+1-self.N)] = Temp_set
        
        #Addig inter_chains latency
        
        for k in range(self.W-1):
            
            for z in range(k*self.d,self.d*(k+1)):
                Temp_set = []
                for y in range((k+1)*self.d,self.d*(k+2)):
                    Temp_set.append(Backup[List[z][self.h-1],List[y][0]])
                data0['Chain'+str(z+1)] = Temp_set                    
        
        #Within chain Latency
            
        for i in range(len(List)):
            data0['Within_Chain'+str(i+1)] = np.sum([ Backup[List[i][j],List[i][j+1]] for j in range(len(List[0])-1)]) 
            
        data0['Mixnodes'] = Backup_1
        data0['List'] = LIST

        return data0

        
    def Random_xx(self,Matrix):
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
        
    
    


    def Random_A(self,Matrix):   
        data0 = {}
        #Adding client to chains laytency
        for j in range(self.N,self.N+self.U):
            Temp_set = []
            for k in range(self.d):
                Temp_set.append(Matrix[k*self.h,j])
            data0['Client'+str(j+1-self.N)] = Temp_set
        
        #Addig inter_chains latency
        
        for k in range(self.W-1):
            
            for z in range(k*self.d,self.d*(k+1)):
                Temp_set = []
                for y in range(k*self.d,self.d*(k+1)):
                    Temp_set.append(Matrix[(z+1)*self.h-1,y*self.h+self.d*self.h])
                data0['Chain'+str(z+1)] = Temp_set                    
        
        #Within chain Latency
            
        for i in range(self.d*self.W):
            data0['Within_Chain'+str(i+1)] = np.sum([Matrix[i*self.h+j,i*self.h+j+1] for j in range(self.h-1)]) 
            
        data0['Mixnodes'] = Matrix
        data0['List'] = [[j for j in range(i*self.h,(i+1)*self.h)]for i in range(self.d*self.W)]
        return data0      
    
    
    
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
   
        self.Sim_2_Sim = [LIST,LIST2]  
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
        LL1 = []
        LL2 = []
        for j in range(self.d):
            LL1.append(data0['Within_Chain'+str(j+1)])
            LL2.append(data0['Within_Chain'+str(self.d+j+1)])
        R1 = np.ones((self.U,self.d))
        R2 = np.ones((self.d,self.d))
        L1 = np.ones((self.U,self.d))
        L2 = np.ones((self.d,self.d))

        for i in range(self.U):
            R1[i,:] = data0['Client'+str(i+1)]
        L1 = R1 + LL1
        for k in range(self.d):
            R2[k,:] = data0['Chain'+str(k+1)]
        L2 = R2 + LL2   
        
        return (R1,L1),(R2,L2)
            
    

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
    def T_entropy(self,Matrix):
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


        Sim = Simulation([List_L,List_R],self.Targets,self.run,self.delay3,self.delay2,self.d,self.h,self.W,self.U)        

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
        with open('data0.json','w') as json_file:
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
            Matrix1 = np.copy(Matrix)
            data1 = self.LONA(Matrix)

            data2 = self.Random_A(Matrix1)

            (L1,L_1),(L2,L_2) = self.Decod_LONA(data1)

            (d1,d_1),(d2,d_2) = self.Decod_LONA(data2)
            data_ = {}
            for item in Names:
                Temp_data1 = []
                Temp_data2 = []
                for Tau in List_Tau:
                    
                    if item == 'LAR' and Tau==0:
                        Tau = 0.1
                    
                    R1 = self.M_Routing(eval('self.'+item),L1,Tau)
                    R2 = self.M_Routing(eval('self.'+item),L2,Tau)                

                    r1 = self.M_Routing(eval('self.'+item),d1,Tau)
                    r2 = self.M_Routing(eval('self.'+item),d2,Tau) 

                    Temp_data1.append([[L_1,L_2],[d_1,d_2]])

                    Temp_data2.append([[R1,R2],[r1,r2]])
                data_[item] = [Temp_data1,Temp_data2]

            data_['Mixnodes'] = [data1['Mixnodes'],data2['Mixnodes']]
            data_['List'] = [data1['List'],data2['List']]           
            data0['It'+str(It+1)] = data_
            
        with open('Results/Basic_data_1.json','wb') as json_file:
                pickle.dump(data0,json_file)
                
        
    def Latency_Entropy(self,T_List,Iterations):
        Names = ['LAR','LAS','GPR','EXP']
        with open('Results/Basic_data_1.json','rb') as json_file:
            data0 = pickle.load(json_file)
        data = {}
        
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
            E_W_LONA_Noise_ = []
            E_WW_LONA_Noise_ = []  
            E_W_LONA_Noise_B_ = []
            E_WW_LONA_Noise_B_ = []                
            Greedy_Balance_E_ = []
            Greedy_Balance_L_ = []
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
                E_W_LONA_Noise = []
                E_WW_LONA_Noise = []
                E_W_LONA_Noise_B = []
                E_WW_LONA_Noise_B = []                
                Greedy_Balance_E = []
                Greedy_Balance_L = []
                

                for It in range(1,Iterations+1):

                    #Latency of LONA for one and two wings setting
                    L1 = self.ave_L(data0['It'+str(It)][item][1][i][0][0],data0['It'+str(It)][item][0][i][0][0])
                    L2 = self.ave_L(data0['It'+str(It)][item][1][i][0][1],data0['It'+str(It)][item][0][i][0][1]) 
                    #Latency of Random for one and two wings setting
                    L_1 = self.ave_L(data0['It'+str(It)][item][1][i][1][0],data0['It'+str(It)][item][0][i][1][0])
                    L_2 = self.ave_L(data0['It'+str(It)][item][1][i][1][1],data0['It'+str(It)][item][0][i][1][1])
                    
                    L_W_LONA.append(L1)
                    L_WW_LONA.append(L1+L2)
                    L_W_Random.append(L_1)
                    L_WW_Random.append(L_1+L_2)
                    
                    #Entropy of LONA for one and two wings setting
                    R1 = self.T_entropy(data0['It'+str(It)][item][1][i][0][0])
                    R2 = self.T_entropy(self.T_matrix([data0['It'+str(It)][item][1][i][0][0],data0['It'+str(It)][item][1][i][0][1]]))
                    #Entropy of Random for one and two wings setting
                    R_1 = self.T_entropy(data0['It'+str(It)][item][1][i][1][0])
                    R_2 = self.T_entropy(self.T_matrix([data0['It'+str(It)][item][1][i][1][0],data0['It'+str(It)][item][1][i][1][1]]))                   

                    E_W_LONA.append(R1)
                    E_WW_LONA.append(R2)
                    E_W_Random.append(R_1)
                    E_WW_Random.append(R_2)
                    #Balance the R1 AND R2 for LONA
                    B1 = Balance_E(data0['It'+str(It)][item][1][i][0][0],self.DP)
                    B2 = Balance_E(data0['It'+str(It)][item][1][i][0][1],self.DP)
                    #Latency of LONA for one and two eings setting
                    L1_B = self.ave_L(B1,data0['It'+str(It)][item][0][i][0][0])
                    L2_B = self.ave_L(B2,data0['It'+str(It)][item][0][i][0][1])                    

                    #Entropy of LONA for one and two eings setting when balancing the routings
                    R1_B = self.T_entropy(B1)
                    R2_B = self.T_entropy(self.T_matrix([B1,B2])) 
                    
                    L_W_LONA_B.append(L1_B)
                    L_WW_LONA_B.append(L1_B+L2_B)
                    E_W_LONA_B.append(R1_B)
                    E_WW_LONA_B.append(R2_B)                    
                    
                    
                    #Adding noise to the rouitng distributions for LONA
                    N1 = self.Noise(data0['It'+str(It)][item][1][i][0][0],0.5)
                    N2 = self.Noise(data0['It'+str(It)][item][1][i][0][1],0.5)                    

                    R1_N = self.T_entropy(N1)
                    R2_N = self.T_entropy(self.T_matrix([N1,N2])) 
                    E_W_LONA_Noise.append(R1_N)
                    E_WW_LONA_Noise.append(R2_N)
                    
                    
                    #Adding noise to the rouitng distributions for LONA
                    NN1 = self.Noise(B1,0.5)
                    NN2 = self.Noise(B2,0.5)                    

                    R1_NN = self.T_entropy(NN1)
                    R2_NN = self.T_entropy(self.T_matrix([NN1,NN2])) 
                    E_W_LONA_Noise_B.append(R1_NN)
                    E_WW_LONA_Noise_B.append(R2_NN)

                    
                    
                E_W_LONA_.append(np.mean(E_W_LONA))
                E_WW_LONA_.append(np.mean(E_WW_LONA))
                E_W_Random_.append(np.mean(E_W_Random))
                E_WW_Random_.append(np.mean(E_WW_Random))
                
                L_W_LONA_.append(np.mean(L_W_LONA))
                L_WW_LONA_.append(np.mean(L_WW_LONA))
                L_W_Random_.append(np.mean(L_W_Random))
                L_WW_Random_.append(np.mean(L_WW_Random)) 
                E_W_LONA_B_.append(np.mean(E_W_LONA_B))
                E_WW_LONA_B_.append(np.mean(E_WW_LONA_B))
                L_W_LONA_B_.append(np.mean(L_W_LONA_B))
                L_WW_LONA_B_.append(np.mean(L_WW_LONA_B))  
                E_W_LONA_Noise_.append(np.mean(E_W_LONA_Noise))
                E_WW_LONA_Noise_.append(np.mean(E_WW_LONA_Noise))
                E_W_LONA_Noise_B_.append(np.mean(E_W_LONA_Noise_B))
                E_WW_LONA_Noise_B_.append(np.mean(E_WW_LONA_Noise_B))                
                
                Greedy_Balance_E_.append(np.mean(Greedy_Balance_E))
                Greedy_Balance_L_.append(np.mean(Greedy_Balance_L))
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
  
            data[item]['E_W_LONA_Noise'] = E_W_LONA_Noise_
            data[item]['E_WW_LONA_Noise'] = E_WW_LONA_Noise_ 

            data[item]['E_W_LONA_Noise_B'] = E_W_LONA_Noise_B_
            data[item]['E_WW_LONA_Noise_B'] = E_WW_LONA_Noise_B_
            
            data[item]['Greedy_E'] = Greedy_Balance_E_
            data[item]['Greedy_L'] = Greedy_Balance_L_
        
        with open('Results/Basic_Latency_Entropy.json','w') as json_file:
            json.dump(data,json_file)   
    
    
    
    
    
    
    def Noise_Latency_Entropy(self,N_List,Iterations):
        Names = ['LAR','LAS','GPR','EXP']
        with open('Results/Basic_data_1.json','rb') as json_file:
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

        with open('Results/Basic_data_1.json','rb') as json_file:
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
            
        with open('D:/Cascades/Results/FCP_Adversary.json','w') as json_file:
            json.dump(data,json_file)
                
                
                
                
                
                
                
                                    
                
                
            
    


    def Adversary_Budget_(self,B_List,Iterations):
        
        Names = ['Random','Adaptive']
        with open('Results/Basic_data_1.json','rb') as json_file:
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


        with open('D:/Cascades/Results/FCP_Adversary_B.json','w') as json_file:
            json.dump(data,json_file)
                
    
    def Baseline_Sim(self,T_List,nn,Iterations):
        Names = ['LAR','GPR','EXP']

        data = {}

        with open('Results/Basic_data_1.json','rb') as json_file:
            data0 = pickle.load(json_file)          
        for name in Names:
            data[name] = {}
        for item in Names:

            E_0= []
            L_0= []

            for i in range(len(T_List)):

                E_1=[]
                L_1=[]


                for It in range(1,Iterations+1):

                    #Latency of LONA for one and two wings setting
                    R1 = [data0['It'+str(It)][item][1][i][0][0],data0['It'+str(It)][item][1][i][0][1]]
                    L1 = [data0['It'+str(It)][item][0][i][0][0],data0['It'+str(It)][item][0][i][0][1]]

    
                    Ave_L, Ave_E = self.Sim(L1, R1, nn)


                    L_1.append(Ave_L[0])

                    
                    #Entropy of LONA for one and two wings setting

                    E_1.append(Ave_E[0])



                    
                E_0.append(E_1)
                L_0.append(L_1)

            data[item]['E'] = E_0
            data[item]['L'] = L_0                   
  

        

            with open('D:/Cascades/Results/Basic_Sim.json','w') as json_file:
                json.dump(data,json_file)
                               
    

    
    
    
    def Baseline_Sim_T(self,T_List,nn,Iterations):

        Names = ['GPR']

        data = {}

        with open('Results/Basic_data_1.json','rb') as json_file:
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
      


    def E2E(self,e2e,T_List,nn,Iterations,name):

        Names = [name]

        data = {}

        with open('Results/Basic_data_1.json','rb') as json_file:
            data0 = pickle.load(json_file)          
        for name in Names:
            data[name] = {}
        for item in Names:

            E_0= []
            L_0= []
            AE_0 = []
            AL_0 = []
            for i in range(len(T_List)):

                E_1=[]
                L_1=[]
                AE_1 = []
                AL_1 = []

                

                for It in range(1,Iterations+1):
                    #Latency of LONA for one and two wings setting
                    L_x = self.ave_L(data0['It'+str(It)][item][1][i][0][0],data0['It'+str(It)][item][0][i][0][0])
                    AL_1.append(L_x)
                    self.delay1 = (e2e-L_x)/3
                    #Entropy of LONA for one and two wings setting
                    AE_1.append(self.T_entropy(data0['It'+str(It)][item][1][i][0][0]))
                    #Latency of LONA for one and two wings setting
                    R1 = [data0['It'+str(It)][item][1][i][0][0],data0['It'+str(It)][item][1][i][0][1]]
                    L1 = [data0['It'+str(It)][item][0][i][0][0],data0['It'+str(It)][item][0][i][0][1]]

    
                    Ave_L, Ave_E = self.Sim(L1, R1, nn)

                    L_1.append(Ave_L[0])

                    
                    #Entropy of LONA for one and two wings setting

                    E_1.append(Ave_E[0])



                    
                E_0.append(E_1)
                L_0.append(L_1)
                AE_0.append(AE_1)
                AL_0.append(AL_1)

            data[item]['E'] = E_0
            data[item]['L'] = L_0                   
            data[item]['AE'] = AE_0
            data[item]['AL']  = AL_0  


            with open('Results/Basic_E2E_'+name+'.json','w') as json_file:
                json.dump(data,json_file)
                               

  
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
  
    

    def Noise_EXP_x(self,N_List,Iterations,i):
        self.c_f = 0.2
        Names = ['LAR','LAS','GPR','EXP']

        with open('Results/Basic_data_1.json','rb') as json_file:
            data0 = pickle.load(json_file)

        data = {}

        
        for name in Names:
            data[name] = {}
        for item in Names:

            E_ = []
            F_ = []

            for noise in N_List:

                E = []
                F = []
                

                for It in range(1,Iterations+1):

                    #Adding noise to the rouitng distributions for LONA
                    N1 = self.Noise(data0['It'+str(It)][item][1][i][0][0],noise)
                    E.append(self.T_entropy(N1))
                    R_List = [N1,N1]
                    Path,Path_ = self.Adaptive_Adv(R_List, self.c_f) 

                    F.append(Path)

                    
                    
                E_.append(np.mean(E))
                F_.append(np.mean(F))
                
                
                

            
            data[item]['E'] = E_
            data[item]['F'] = F_
        

        with open('Results/Noise_EXP.json','w') as json_file:
            json.dump(data,json_file)   
    
























