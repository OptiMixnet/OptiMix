# OptiMix

This repository contains the artifact for the paper titled "OptiMix: Scalable and Distributed Approaches for Latency Optimization in Modern Mixnets," accepted for publication at NDSS 2026.



## Initial setup and dependencies
To execute the artifact, the code can be run on any standard laptop or workstation using Ubuntu 18.04 or higher. It is compatible with Python versions from 3.8.10 up to (but not including) 3.11. Importantly, the artifact includes precisely the same configurations and settings as those used in the original OptiMix evaluation. The only exception is that the number of iterations has been scaled down to ensure feasibility on standard hardware.

The artifact has been optimized to run on systems with at least 16\,GB of RAM and 50\,GB of available disk space. These specifications allow users to reproduce results efficiently without requiring access to high-performance computing environments.

Before executing the code, please ensure your system satisfies the following requirements: Ubuntu 18.04 or higher, Python version between 3.8.10 and 3.10.x (not 3.11 or newer), a minimum of 16\,GB of RAM, and at least 50\,GB of free disk space.

All required dependencies for execution are listed in the \texttt{dependencies.txt} file included in the repository.


- matplotlib==3.5.2
- numpy==1.21.2
- plotly==5.10.0
- pulp==2.7.0
- scikit_learn==1.1.1
- scikit-learn-extra==0.2.0
- scipy==1.8.1
- simpy==4.0.1
These dependencies can be easily installed using the following command:
\begin{verbatim}
pip install -r dependencies.txt
\end{verbatim}


## How to Use
Describe steps here...

## License
MIT or your preferred license.
