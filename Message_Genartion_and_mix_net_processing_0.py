# -*- coding: utf-8 -*-
"""
This Python file, on behalf of clients, generates the messages to be sent to the mixnet for cascade (butterfly) topologies.
"""
import math
import numpy as np
from scipy.stats import expon
from Message_ import message




class Message_Genartion_and_mix_net_processing(object):
    
    def __init__(self,env,Mixes,capacity,MNet,num_target,delay,d,h,W,U,nn):
        
        self.env = env
        
        self.Mixes = Mixes
        
        self.capacity = capacity
        
        #self.Latency_List = Dict_List[0]
        
        #self.Routing_List = Dict_List[1]
        
        self.MNet = MNet
        
        self.NT = num_target
        
        self.delay = delay
        self.Wings = W
        self.hops = h
        self.depth = d
        self.U = U
        self.N = self.depth*self.hops*self.Wings
        self.nn = nn


    def Prc(self):

        
        #This function is written to be used for generating the messages through the mix network

        ID = 0 #The id of the first target messages
        for i in range(self.nn*self.depth):#generate a fix number of messages to initiate the network
            TARGET = False
            target_id = -1
       
            client_id = 'Cl' 
            M = message('Message%02d' %i,self.NT,target_id,client_id)#message is being created

            self.env.process(self.MNet.Message_Traveling(M))#message send to the mix net

        i = self.nn*self.depth
        while True:# Create other messages by an exponential delay
            t2 = expon.rvs(scale=self.delay)#The exponential delay between two succeeding messages
            yield self.env.timeout(t2)
            

            TARGET = False
            target_id = -1
            if(ID < self.NT):
                y = np.random.multinomial(1, [1/2,1/2], size=1)[0][0]
                if y==1:
                    TARGET = True
                if TARGET:
                    target_id = ID
                    ID = ID + 1
                    
            
            client_id = 'Cl' 
            M = message('Message%02d'%i,self.NT,target_id,client_id)
            self.env.process(self.MNet.Message_Traveling(M))
                
            i = i + 1









