# -*- coding: utf-8 -*-
"""
This file helps generate some initial datasets needed for the running OptiMix using both the Nym and RIEP datasets. It is also useful for reproducing Table 3.
"""
import pickle
import numpy as np
from data_set import Dataset
from data_refine import DNA

def Latency_extraction(data0,Positions,L):
    List = []
    n1,n2 = np.shape(data0)
    for i in range(L-1):
        List_ = []
        for j in range(int(n1/L)):
            List__ = []
            for k in range(int(n1/L)):
                List__.append(data0[Positions[i][j],Positions[i+1][k]])
            List_.append(List__)
        List.append(List_)
    return List

def Norm_List(List,term):
    S = np.sum(List)
    return [List[i]*(term/S)for i in range(len(List))]

class data_MixNet(object):
    
    def __init__(self,W1,W2,L,Iterations,make_data = False):

        self.W1 = W1
        self.W2 = W2
        self.L = L
        self.N1 = self.W1*self.L
        self.N2 = self.W2*self.L
        self.b = 2
        self.WW = {'NYM':self.W1,'RIPE':self.W2}
        self.Iterations = Iterations

        self.Data_Set_General = {'NYM':{},'RIPE':{}}
        
        if make_data:
            self.data_generator(self.Iterations)
            
            

    def data_generator(self,Iteration):
        corrupted_Mix = {}
        for i in range(9):
            corrupted_Mix['PM%d' %i] = False        

        
        data0 = {}
        W = {'NYM':self.W1,'RIPE':self.W2}
        
        for Data in ['NYM','RIPE']:
            data_class = Dataset(W[Data], self.L)
            data1 = {}
            for It in range(Iteration):

                
                LATLON, Cart, Matrix_,Omega_ = eval('data_class.' + Data+ '()')
                Matrix = np.copy(Matrix_)
                Omega = Omega_.copy()
                #########DNA#########################################################
                Class_DNA = DNA(Omega,Matrix,self.L,self.b)    
                Positions,_ = Class_DNA.DNA_Arrangement_W()
                
                Latency_List = Latency_extraction(Matrix, Positions, self.L)
                O_ = [Norm_List(item,W[Data]) for item in _]
                
                data4 = {'Latency_List': Latency_List,'Omega':O_, 'Positions':Positions,'Loc':np.transpose(Cart),'Matrix':Matrix,'x':Omega_,'xx':_}

                data1['It'+str(It+1)] = {'DNA':data4}
                
            data0[Data] = data1
            
            
        

        with open('data0.pkl','wb') as pkl_file:
            pickle.dump(data0,pkl_file)
          


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    