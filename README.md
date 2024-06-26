### Optimal Maritime Connectivity | Algorithmic Modeling of Liner Shipping Networks

By August Høgsted and Johan Matzen Brodt

Supervised by Stefan Røpke
<br>

__Keywords:__ LINER-LIB, Liner Shipping Network Design Problem, Simulated Annealing metaheuristic, Dantzing Wolfe decomposition, Column generation, Service design, MIP, optimization.

<br>
This project proposes a method for solving the Liner Shipping Network Design Problem (LSNDP). The LSNDP presents intricate challenges in optimizing shipping routes and cargo flow.

The proposed method employs a two-stage methodology, comprising service-first and flow-second algorithms. The service-first algorithm involves a simulated annealing (SA) metaheuristic to construct services and minimize their fixed costs, functioning as the foundational network in the second stage. The flow-second algorithm involves a Mixed-Integer Program (MIP) for solving the LSNDP. The models are evaluated on the Baltic, West Africa, and Mediterranean instances from LINER-LIB which is a benchmark suite for model testing. The LINER-LIB documentation can be found here: http://www.linerlib.org.

The 'Results' folder encompasses the resulting services of the SA metaheuristic along with the output of the MIP for each instance. The implementatation of the SA metaheuristic can be found in Service-Network-Design.py and the MIP models are found in flowBaltic_final.ipynb, flowWAF_final.ipynb, and flowMed_final.ipynb. The .jpg files illustrate the included services in the solutions of the MIPs.
