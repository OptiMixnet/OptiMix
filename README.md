# OptiMix

This repository contains the artifact for the paper titled "OptiMix: Scalable and Distributed Approaches for Latency Optimization in Modern Mixnets," accepted for publication at NDSS 2026.



## Initial setup and dependencies
To execute the artifact, the code can be run on any standard laptop or workstation using Ubuntu 18.04 or higher. It is compatible with Python versions from 3.8.10 up to (but not including) 3.11. Importantly, the artifact includes precisely the same configurations and settings as those used in the original OptiMix evaluation. The only exception is that the number of iterations has been scaled down to ensure feasibility on standard hardware.

The artifact has been optimized to run on systems with at least 16\,GB of RAM and 50\,GB of available disk space. These specifications allow users to reproduce results efficiently without requiring access to high-performance computing environments.

Before executing the code, please ensure your system satisfies the following requirements: Ubuntu 18.04 or higher, Python version between 3.8.10 and 3.10.x (not 3.11 or newer), a minimum of 16\,GB of RAM, and at least 50\,GB of free disk space.

All required dependencies for execution are listed in the `dependencies.txt` file included in the repository.


- matplotlib==3.5.2
- numpy==1.21.2
- plotly==5.10.0
- pulp==2.7.0
- scikit_learn==1.1.1
- scikit-learn-extra==0.2.0
- scipy==1.8.1
- simpy==4.0.1
  
These dependencies can be easily installed using the following command: `pip install -r dependencies.txt`



## Code execution

Please refer to the `OptiMix_artifact.pdf` for detailed descriptions of each experiment and the claims they support. To execute the code, run: `python3 Main.py`
At the beginning, you must run `Main.py` with `Input = 0` once and for all. This initializes the required files and datasets necessary for executing all subsequent experiments.
To run the rest of the artifact suite, set the `Input` argument as described below to execute specific experiments or to generate any individual figures or tables:




- E1:  `Input = 1`
- E2:  `Input = 2`
- E3:  `Input = 3`

- Fig. 2: `Input = 22`
- Fig. 3: `Input = 33`
- Fig. 4: `Input = 44`
- Fig. 5: `Input = 55`
- Fig. 6: `Input = 66`
- Fig. 7: `Input = 77`
- Fig. 8: `Input = 88`
- Fig. 9: `Input = 99`
  
- Tab. 1: `Input = 100`
- Tab. 2: `Input = 200`
- Tab. 3: `Input = 300`

## Additional Notes

- After running each experiment, the corresponding figures will be automatically saved in the "Figures" folder, and the corresponding tables in the "Tables" folder. In case LaTeX is not installed, table results will be printed directly in the terminal.
    
- For each experiment, we provide initial values for the number of iterations in `config.py` to ensure reproducibility of results similar to those in the paper. These values can be modified as needed. Specifically, increasing the number of iterations improves accuracy and reduces sampling errors, but also increases execution time. Hence, we set the default number of iterations to five.

- If the following warnings appear during execution, you can safely ignore them:
       
1) RuntimeWarning: Mean of empty slice. out=out, **kwargs

2) invalid value encountered in scalar divide ret = ret.dtype.type(ret / rcount)

## Hardware Requirements
The code is tested to run on commodity hardware with 16 GB RAM, 8 cores, and 50 GB hard disk storage.

## Description of Files in the Repository

The artifact repository contains the following main files:

-	`CLUSTER.py`: Clustering is the first step in LARMix [29], used for optimizing latency.
  
-	`config.py`: This file is the config file, containing all the variables and their values needed to initialize the functions required for running the experiments. You can adjust the variables as needed.
  
- `Data_Gen.py`: This file helps generate some initial datasets needed for the running OptiMix using both the Nym and RIEP datasets. It is also useful for reproducing Table 3.  
- `data_refine.py`: This .py file helps adapt some of the data interfaces used in OptiMix.
- `data_set.py`: Datasets is a class for creating a raw dataset from Nym or RIPE, making it usable for emulating 
the link delays between mix nodes.  
- `dependencies.txt`: Provides the necessary script to install the dependencies. 
- `Fancy_Plot.py`: This class or set of functions provides all the necessary code for generating the plots used in OptiMix. 
- `FCP_Functions.py`: This .py file includes strategies that a mixnode adversary might consider when corrupting mindnodes 
in mixnets with stratified topologies. 
- `FCP_Functions_.py`: This .py file includes strategies that a mixnode adversary might consider when corrupting mindnodes 
in mixnets with cascade topologies.
- `GateWay.py`: Emulates the clients connecting to the mixnets.
- `Greedy_LARMIX.py`: Provides the Greedy algorithm in  LARMix [29].  
- `LARMix_Greedy.py`: Provides the Greedy algorithm in  LARMix [29].
- `Main.py`: This file provides instructions regarding how to run the experiments described in the main body of the paper. 
- `Main_F.py`: Main_F.py contains a comprehensive set of functions for analyzing OptiMix on stratified topologies, including all necessary functions to compute anonymity and latency metrics.
Additionally, this file includes the main functions required to reproduce the key results of LARMix [29] and LAMP [30]. 
- `Main_F_.py`: Main_F.py contains a comprehensive set of functions for analyzing OptiMix on cascade (also butterfly) topologies, including all necessary functions to compute anonymity and latency metrics. 
- `Message_.py`: Simulates the messages generated and sent by the clients.  
- `Message_Genration_and_mix_net_process0.py`:  This Python file, on behalf of clients, generates the messages to be sent to the mixnet for cascade (butterfly) topologies.
- `Message_Genration_and_mix_net_process_.py`: This Python file, on behalf of clients, generates the messages to be sent to the mixnet for stratified topologies.
- `Message_Genration_and_mix_net_process_P.py`: This Python file, on behalf of clients, generates the messages to be sent to the mixnet 
for the prior works LARMix [29] and LAMP [30].
- `Mix_Node_.py`: Simulates using discrete event simulation, a mixnode in mixnets.
- `MixNetArrangement.py`: This .py file implements the diversification algorithm used in LARMix [29], which helps arrange the mixnodes into layers. 
- `NYM.py`: This .py file provides the main simulation components necessary to simulate a stratified mixnet as used in OptiMix.
- `NYM_.py`: Mix_Net: This .py file provides the main simulation components necessary to simulate a cascade (butterfly) mixnet as used in OptiMix. 
- `Nym_data.json`: The Nym dataset used in the experiments.
- `Nym_dataset.json`: The Nym dataset used in the experiments (version 2).
- `NYM_P.py`: Mix_Net: This .py file provides the main simulation components necessary to simulate 
  mixnets used in prior works LARMix [29] and LAMP [30].
- `OptiMix.py`: This .py file includes all the functions needed to directly run the main experiments of OptiMix 
and generates the corresponding figures.
- `Ripe_dataset.json`: The RIPE dataset used in the experiments.
- `Routings.py`: This function helps to model the routing approaches in the prior works LARMix [29] and LAMP [30].
- `Sim.py`: This .py file also includes the necessary simulation components for reproducing simulations of OptiMix.
- `Sim_P.py`: This .py file also includes the necessary simulation components for reproducing simulations of prior works LARMix [29] and LAMP [30].


## License
MIT or your preferred license.
