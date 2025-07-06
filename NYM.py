# -*- coding: utf-8 -*-
"""
Mix_Net: This .py file provides the main simulation components necessary to simulate a stratified mixnet as used in OptiMix.
"""

import numpy  as np

class MixNet(object):
    #In this class we are gonna make an instantiation of the  mix net
    def __init__(self,env, Mixes,GateWays):
        self.env = env
        #This is agian simpy  environment
        self.M = Mixes
        #List of mixes created from the mix class
        self.GW = GateWays ##List of gate ways from the GW class
        self.N = len(self.M) #Number of all mix nodes
        self.W = int(self.N/3)#Number of mix nodes in each mixing layer
        self.G = len(self.GW)
        self.LL = []
        self.LT = []
        #End_to_end latency are added to this list
        self.EN =[]
        # Entropy or distribution are appended to this list for each individual message
    def Message_Traveling(self,message,Mix_Dict):
        #This is the main function to help messages get routed in a mix net, It should
        #receive the messages and mix dictionaries.

        w = self.W
        

        Pro = np.random.multinomial(1, [1/self.G]*self.G, size=1)[0]
        GG = 'G' + str(Pro.tolist().index(1)+1)


        G =GG
        
        G1 = self.GW[GG]

        message.mtime.append(self.env.now)#The time in which message enter to the network should mark
        if not  message.target_id == -1:
            message.Ttime.append(self.env.now)      
        yield self.env.process(G1.Receive_and_send(message))
        

        
        Pro = np.random.multinomial(1, Mix_Dict[G][1], size=1)

        #First mix node should be chosen randomly.
        #here a realization of the uniform distribution will be considered
        j = -1
        for item in Pro[0]:
            j = j +1
            if item == 1:
                Next = j
        PM1 = 'PM%d' %(Next+1)
                

        GW_delay2 = Mix_Dict[G][0][Next]

        yield self.env.timeout(GW_delay2) 
     
        M1 = self.M[Next]#The mix will be selected upon making the realization

        yield self.env.process(M1.Receive_and_send(message))#Then it should wait to be accepted by the mix node
        # The next hop should be chosen in accordance with the biased distribution
        m1 = Next
        ll = Next +1
        Pro = np.random.multinomial(1, Mix_Dict['PM%d' %ll][1], size=1)
        kk = -1
        for item in Pro[0]:
            kk = kk+1
            if item == 1:
                Next = kk +w
                break
        yield self.env.timeout(Mix_Dict[PM1][0][Next-w])# Then link delay will be yielded
        #The rest of the code is similar just try to route the message
        M2 = self.M[Next]
        yield self.env.process(M2.Receive_and_send(message))
        ll = Next +1
        Pro = np.random.multinomial(1, Mix_Dict['PM%d' %ll][1], size=1)
        m2 = Next
        kk = -1
        for item in Pro[0]:
           
            kk = kk+1
            if item == 1:
                Next = kk +2*w
                break
        yield self.env.timeout(Mix_Dict['PM%d' %ll][0][Next-2*w])
        M3 = self.M[Next]
        yield self.env.process(M3.Receive_and_send(message))

        
      
        message.mtime.append(self.env.now)#The time of leaving the mix net should be written down
        
        self.LL.append(message.mtime[1]-message.mtime[0])#The latency will be added to the latency list
        if not  message.target_id == -1:
            message.Ttime.append(self.env.now)
            self.LT.append(message.mtime[1]-message.mtime[0])            
        self.EN.append(message.prob)#The message dist will be added to the entropy list
       
       
       




