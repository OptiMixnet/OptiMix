# -*- coding: utf-8 -*-
"""
Mix_Net: This .py file provides the main simulation components necessary to simulate a cascade (butterfly) mixnet as used in OptiMix.
"""


import numpy  as np


def To_list(matrix):
    """Convert a numpy matrix to a list. If the result is a single row, return it as a flat list."""
    matrix_list = matrix.tolist()
    return matrix_list[0] if len(matrix_list) == 1 else matrix_list



class MixNet(object):
    #In this class we are gonna make an instantiation of the  mix net
    def __init__(self,env, Mixes,Dict_List,d,h,W,U):
        self.env = env
        #This is agian simpy  environment
        self.M = Mixes
        self.depth = d
        self.hops = h
        self.Wings = W
        self.U = U
        #List of mixes created from the mix class
        self.N = self.Wings*self.hops*self.depth
        self.LL = []
        self.LL0 = []
        #End_to_end latency are added to this list
        self.EN0 = []
        self.EN =[]
        self.Latency_List = Dict_List[0]
        self.Routing_List = Dict_List[1]
        
        # Entropy or distribution are appended to this list for each individual message
    def Message_Traveling(self,message):
        #This is the main function to help messages get routed in a mix net, It should
        #receive the messages and mix dictionaries.


        message.mtime.append(self.env.now)#The time in which message enter to the network should mark
        
        Client_ID = round(self.U*(np.random.rand(1)[0]))#messages should be assigned to a client
        
        if Client_ID == self.U:
            Client_ID = self.U-1



        Ch1 = To_list(np.random.multinomial(1, To_list(self.Routing_List[0][Client_ID,:]), size=1)).index(1)
        Chain1 = [self.M[Ch1*self.hops+itt] for itt in range(self.hops)]#The first chain will be selected upon making the realization

        self.env.timeout(To_list(self.Latency_List[0])[Client_ID][Ch1])
        
        for itt in range(self.hops):
            yield self.env.process(Chain1[itt].Receive_and_send(message))#Mixing delay
            
        self.LL0.append(self.env.now-message.mtime[0])#The latency will be added to the latency list
            
        self.EN0.append(message.prob)#The message dist will be added to the entropy list
               
        
        
        Ch2 = To_list(np.random.multinomial(1, To_list(self.Routing_List[1][Ch1,:]), size=1)).index(1)
        Chain2 = [self.M[(Ch2)*self.hops+itt+self.depth*self.hops] for itt in range(self.hops)]#The first chain will be selected upon making the realization
        
        self.env.timeout(To_list(self.Latency_List[1])[Ch1][Ch2])
        for itt in range(self.hops):
            yield self.env.process(Chain2[itt].Receive_and_send(message))#Mixing delay
                    

    
            
        
        message.mtime.append(self.env.now)#Exit time
                
        
        
        self.LL.append(message.mtime[1]-message.mtime[0])#The latency will be added to the latency list
            
        self.EN.append(message.prob)#The message dist will be added to the entropy list
       
       
       




