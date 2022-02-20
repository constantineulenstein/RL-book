from typing import Tuple, Callable, Sequence
from rl.chapter11.control_utils import glie_mc_finite_learning_rate_correctness
from rl.chapter11.control_utils import \
    q_learning_finite_learning_rate_correctness
from rl.chapter11.control_utils import \
    glie_sarsa_finite_learning_rate_correctness
from rl.chapter11.control_utils import compare_mc_sarsa_ql
from rl.chapter3.simple_inventory_mdp_cap import SimpleInventoryMDPCap
from rl.monte_carlo import glie_mc_control_tabular
from rl.approximate_dynamic_programming import NTStateDistribution
from rl.distribution import Categorical


capacity: int = 2
poisson_lambda: float = 1.0
holding_cost: float = 1.0
stockout_cost: float = 10.0

si_mdp: SimpleInventoryMDPCap = SimpleInventoryMDPCap(
    capacity=capacity,
    poisson_lambda=poisson_lambda,
    holding_cost=holding_cost,
    stockout_cost=stockout_cost
)


gamma: float = 0.9
mc_episode_length_tol: float = 1e-5
num_episodes = 10000

epsilon_as_func_of_episodes: Callable[[int], float] = lambda k: 1/k

nt_states = si_mdp.non_terminal_states
state_distribution: NTStateDistribution = Categorical({state: 1/len(nt_states) for state in nt_states})



i = 0
for episode_qvalue_function in glie_mc_control_tabular(si_mdp,state_distribution,gamma, epsilon_as_func_of_episodes):
    i+=1
    if i ==10000:
        break

print(episode_qvalue_function)
for state in episode_qvalue_function:
    dictionary = episode_qvalue_function[state]
    action = max(dictionary, key=dictionary.get)
    print(f"{state}: {action}")



#print(sorted(episode_value_function.items(), key=lambda x: (x[0].state.on_hand, x[0].state.on_order)))