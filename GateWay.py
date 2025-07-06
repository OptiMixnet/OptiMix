# -*- coding: utf-8 -*-
"""
Emulates the clients connecting to the mixnets.
"""


class GateWay(object):
    def __init__(self,env,name,Processing_delay):
       # A mix node instantiation may have different attributes we described them
        #as follows.
        self.env =env    
        #1)environment: which simply passes the environment of discrete event simulation,
        self.name = name
        #2)name: which is associated with the name of mixes starting from 00 to N-1,

        self.delay = Processing_delay


        #6)Target probability actually refers to the probability of mix node which is computed by
    #summing over all messages distributions it should be zero at the first time as it has no message.
        self.pool = 0
        #The number of messages currently stays in the mix node
   
    def Receive_and_send(self,message):
        

        self.pool = self.pool + 1
        # Upon leting a message in number of messages in the pool will be added by one.
           
        #If the mix node is not corrupted the probability of the message and the mix
        #should be updated.(The probability of the message will be added to the mix node)

                   
               

           
         # The message will lie in the mix node for a random exponential time
        t1 = self.delay
        yield self.env.timeout(t1)

                   
        #The number of messages will be reduced
        self.pool = self.pool - 1
           
