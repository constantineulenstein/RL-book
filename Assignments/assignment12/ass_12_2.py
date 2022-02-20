from rl.chapter2.simple_inventory_mrp import SimpleInventoryMRPFinite
from typing import Iterable, TypeVar
from rl.markov_process import TransitionStep
import itertools

S = TypeVar('S')
A = TypeVar('A')

user_capacity = 2
user_poisson_lambda = 1.0
user_holding_cost = 1.0
user_stockout_cost = 10.0
user_gamma = 0.9
si_mrp = SimpleInventoryMRPFinite(
    capacity=user_capacity,
    poisson_lambda=user_poisson_lambda,
    holding_cost=user_holding_cost,
    stockout_cost=user_stockout_cost
)
si_mrp.display_value_function(gamma=user_gamma)


from rl.distribution import Choose
from rl.td_lambda import td_lambda_tabular



traces: Iterable[Iterable[TransitionStep[S]]] = \
        si_mrp.reward_traces(Choose(si_mrp.non_terminal_states))

i = 0
for episode_value_function in td_lambda_tabular(traces, user_gamma, lambd=0.3):
    i+=1
    if i ==6000000:
        break

print(sorted(episode_value_function.items(), key=lambda x: (x[0].state.on_hand, x[0].state.on_order)))
