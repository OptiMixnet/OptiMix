# -*- coding: utf-8 -*-
"""
This file is the config file, containing all the variables and their values needed to
initialize the functions required for running the experiments. You can adjust the variables as needed.

However, please note:
Not all configuration parameters in the config file can be freely changed. 
Modifying some of them may require a deeper understanding of the methodology and insights from the paper.

For ease and safety, we recommend only changing a few variables such as:

Iterations

Num_targets

run_time
"""



# Mixnets structures
d_Nym  =  80   #Number of nodes in each layer of the mixnet when instantiating OptiMix with the Nym dataset (note: the maximum is 80).

d_RIPE = 200  #Number of nodes in each layer of the mixnet when instantiating OptiMix with the RIPE dataset (note: the maximum is 200).


hops   = 3     #Number of hops in each mix chain of the mixnet.

wings  = 2    #Number of wings in the mixnet.


layers  = 3    #Number of wings (layers) in the stratified mixnet.




# Simulations paramters

Num_targets = 200 #Number of target messages used for estimating H(m) in the simulation.

run_time = 0.5   # Simulation runtime (in seconds).

delay1 = 0.05 #Average of the exponential probability distribution used for modeling the delay of packets inside each mix node (in seconds).

delay2_Nym = delay1/(5*d_Nym) #Average inter-arrival delay between receiving messages at the input of mix nodes, based on the exponential probability distribution (in seconds, for Nym).

delay2_RIPE = delay1/(5*d_RIPE) #Average inter-arrival delay between receiving messages at the input of mix nodes, based on the exponential probability distribution (in seconds, for RIPE).

Iterations = 5   #Number of iterations.

n_scale = 200  #Scaling factor used in message generation during simulations.



# Inputs of other functions

create_data = True # To generate data needed for the experiments


# Set of the corrupted mixes

corrupted_Mix = {}
for i in range(d_Nym*layers):
    corrupted_Mix['PM'+str(i+1)] = False


corrupted_Mix_ = {}
for i in range(d_Nym*hops*wings):
    corrupted_Mix_['PM'+str(i+1)] = False


# List of values for the tuning parameters \tau and cost parameter \theta and Adversary Budget


Tau_List = [0,0.2,0.4,0.6,0.8,1]

Theta_List =  [0,0.01,0.03,0.05,0.08,0.1]

Adv_Budget_List = [0.1,0.15,0.2,0.25]


#E2E delay constraint

e2e_delay = 0.16


# File names

File_figure = "Figures" # In this file the figures will be saved

File_result = "Results" # In this file data will be saved

File_table = "Tables" # In this file all the generated tables will be saved