from rl.chapter2.simple_inventory_mrp import SimpleInventoryMRPFinite
from typing import Iterable, Iterator, TypeVar, Callable, Dict, Sequence, Mapping
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


from rl.chapter2.simple_inventory_mrp import InventoryState
from rl.function_approx import Tabular
from rl.approximate_dynamic_programming import ValueFunctionApprox
from rl.distribution import Choose
from rl.iterate import last
from rl.monte_carlo import mc_prediction, mc_prediction_tabular
from rl.td import td_prediction_tabular
from itertools import islice
from pprint import pprint


traces: Iterable[Iterable[TransitionStep[S]]] = \
        si_mrp.reward_traces(Choose(si_mrp.non_terminal_states))

i = 0
for episode_value_function in mc_prediction_tabular(traces, user_gamma, 1e-6):
    i+=1
    if i ==1000:
        break

print(sorted(episode_value_function.items(), key=lambda x: (x[0].state.on_hand, x[0].state.on_order)))


transitions: Iterable[TransitionStep[S]] = itertools.chain.from_iterable(
        itertools.islice(episode, 100) for episode in traces)

i = 0
for episode_value_function in td_prediction_tabular(transitions, user_gamma):
    i+=1
    if i ==10000:
        break

print(sorted(episode_value_function.items(), key=lambda x: (x[0].state.on_hand, x[0].state.on_order)))
#
# it: Iterator[ValueFunctionApprox[InventoryState]] = mc_prediction(
#     traces=traces,
#     approx_0=Tabular(),
#     gamma=user_gamma,
#     episode_length_tolerance=1e-6
# )
# num_traces = 60000
# last_func: ValueFunctionApprox[InventoryState] = last(islice(it, num_traces))
# pprint({s: round(last_func.evaluate([s])[0], 3) for s in si_mrp.non_terminal_states})