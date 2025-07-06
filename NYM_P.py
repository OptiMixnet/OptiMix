# -*- coding: utf-8 -*-
"""
Mix_Net: This .py file provides the main simulation components necessary to simulate 
mixnets used in prior works LARMix [29] and LAMP [30].
"""
import numpy  as np

def Norm_List(List,term):
    S = np.sum(List)
    return [List[i]*(term/S)for i in range(len(List))]


class MixNet(object):
    #In this class we are gonna make an instantiation of the  mix net
    def __init__(self,env, Mixes,L):
        self.env = env
        #This is agian simpy  environment
        self.M = Mixes
        self.L = L
        #List of mixes created from the mix class
        self.N = len(self.M) #Number of all mix nodes
        self.W = int(self.N/self.L)#Number of mix nodes in each mixing layer
        self.LL = []
        self.LT = []
        #End_to_end latency are added to this list
        self.EN =[]
        # Entropy or distribution are appended to this list for each individual message
    def Message_Traveling(self,message,Mix_Dict):
        #This is the main function to help messages get routed in a mix net, It should
        #receive the messages and mix dictionaries.


        message.mtime.append(self.env.now)#The time in which message enter to the network should mark
        
        mix_first = (np.random.multinomial(1, Mix_Dict['First'], size=1).tolist()[0]).index(1)
        if mix_first == self.W:
            mix_first = self.W-1




        M1 = self.M[mix_first]#The mix will be selected upon making the realization




        yield self.env.process(M1.Receive_and_send(message))#Mixing delay
        
        
        current_mix = mix_first
        for I in range(self.L-1):

            
            now_mix = (np.random.multinomial(1, Norm_List(Mix_Dict['Routing'][I][current_mix],1), size=1).tolist()[0]).index(1)
            

            yield self.env.timeout(Mix_Dict['Latency'][I][current_mix][now_mix])# Then link delay will be yielded
            #The rest of the code is similar just try to route the message
            current_mix = now_mix
            M_ = self.M[current_mix+(I+1)*self.W]
            yield self.env.process(M_.Receive_and_send(message))
    
            
        
        message.mtime.append(self.env.now)#Exit time
                
        
        
        self.LL.append(message.mtime[1]-message.mtime[0])#The latency will be added to the latency list
        if not  message.target_id == -1:
            message.Ttime.append(self.env.now)
            self.LT.append(message.mtime[1]-message.mtime[0])            
        self.EN.append(message.prob)#The message dist will be added to the entropy list
       
       
       




