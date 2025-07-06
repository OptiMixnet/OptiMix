# -*- coding: utf-8 -*-
"""
Main: This file provides instructions regarding how to run the experiments described in the main body of the paper.
"""


"""
++++++++++++++++++++++++++++++++++++++++Initializations++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

To initialize, please enter 0. This step is required to generate some .json and .pkl files needed for running the experiments.





.............................E1, E2 and E3.......................................................

To run the experiments:

Enter 1 for Experiment E1

Enter 2 for Experiment E2

Enter 3 for Experiment E3  
    
    
-------------------------------------Figures and Tables-------------------------------------------------------------------------

To reproduce any figure or table from the main body of the paper, 
please enter the corresponding input arguments based on the following table:




--------------------------------------------------------------------------------------------
Fig.2: 22, Fig.3: 33, Fig.4: 44, Fig.5: 55, Fig.6: 66, Fig.7: 77, Fig.8: 88, Fig.9: 99

Tab.1: 100, Tab.2: 200, Tab.3: 300

--------------------------------------------------------------------------------------------
"""

from OptiMix import OptiMix



Input = input("Please enter the ID of the experiment you wish to run:                 ")


OptiMix_class = OptiMix(Input)
