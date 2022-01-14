from dataclasses import dataclass
from typing import Tuple, Dict, Mapping
from rl.markov_process import MarkovRewardProcess
from rl.markov_process import FiniteMarkovRewardProcess
from rl.markov_process import State, NonTerminal
from scipy.stats import poisson
from rl.distribution import SampledDistribution, Categorical, \
    FiniteDistribution
import numpy as np



@dataclass(frozen=True)
class SandLState(State):
    position: int

class SnakeAndLaddersMRPFinite(FiniteMarkovRewardProcess[SandLState]):
    def __init__(
            self,
            num_fields: int = 100,
            extra_steps: Dict = None,
            dice_cost: int = -1
    ):
        self.num_fields: int = num_fields
        self.dice_cost: int = dice_cost
        self.extra_steps: Dict = extra_steps

        super().__init__(self.get_transition_reward_map())

    def get_transition_reward_map(self) -> Mapping[SandLState, Categorical[Tuple[SandLState,int]]]:
        d: Dict[SandLState, Categorical[Tuple[SandLState,int]]] = {}

        for field in range(self.num_fields):
            sr_probs_map = {}

            for i in range(1, 7):
                next_state = field + i
                if i + field in list(self.extra_steps.keys()):
                    next_state = self.extra_steps[field + i]
                if i == 6:
                    reward = 0
                else:
                    reward = self.dice_cost

                sr_probs_map[(SandLState(next_state), reward)] = 1 / 6

            d[SandLState(field)] = Categorical(sr_probs_map)
        return d




# class SimpleInventoryMRPFinite(FiniteMarkovRewardProcess[InventoryState]):
#
#     def __init__(
#         self,
#         capacity: int,
#         poisson_lambda: float,
#         holding_cost: float,
#         stockout_cost: float
#     ):
#         self.capacity: int = capacity
#         self.poisson_lambda: float = poisson_lambda
#         self.holding_cost: float = holding_cost
#         self.stockout_cost: float = stockout_cost
#
#         self.poisson_distr = poisson(poisson_lambda)
#         super().__init__(self.get_transition_reward_map())
#
#     def get_transition_reward_map(self) -> \
#             Mapping[
#                 InventoryState,
#                 FiniteDistribution[Tuple[InventoryState, float]]
#             ]:
#         d: Dict[InventoryState, Categorical[Tuple[InventoryState, float]]] = {}
#         for alpha in range(self.capacity + 1):
#             for beta in range(self.capacity + 1 - alpha):
#                 state = InventoryState(alpha, beta)
#                 ip = state.inventory_position()
#                 beta1 = self.capacity - ip
#                 base_reward = - self.holding_cost * state.on_hand
#                 sr_probs_map: Dict[Tuple[InventoryState, float], float] =\
#                     {(InventoryState(ip - i, beta1), base_reward):
#                      self.poisson_distr.pmf(i) for i in range(ip)}
#                 probability = 1 - self.poisson_distr.cdf(ip - 1)
#                 reward = base_reward - self.stockout_cost *\
#                     (probability * (self.poisson_lambda - ip) +
#                      ip * self.poisson_distr.pmf(ip))
#                 sr_probs_map[(InventoryState(0, beta1), reward)] = probability
#                 d[state] = Categorical(sr_probs_map)
#         return d


if __name__ == '__main__':
    snl_steps = {
        1: 38,
        4: 14,
        9: 31,
        16: 6,
        28: 84,
        36: 44,
        47: 26,
        49: 11,
        51: 67,
        56: 53,
        62: 19,
        64: 60,
        71: 91,
        80: 100,
        93: 73,
        95: 75,
        98: 78,
        101: 99,
        102: 78,
        103: 97,
        104: 96,
        105: 75,
    }
    snl_mrp = SnakeAndLaddersMRPFinite(
        num_fields=100,
        extra_steps=snl_steps,
        dice_cost=-1
    )

    user_gamma = 1

    from rl.markov_process import FiniteMarkovProcess
    print("Transition Map")
    print("--------------")
    print(FiniteMarkovProcess(
        {s.state: Categorical({s1.state: p for s1, p in v.table().items()})
         for s, v in snl_mrp.transition_map.items()}
    ))

    print("Transition Reward Map")
    print("---------------------")
    print(snl_mrp)

    # print("Stationary Distribution")
    # print("-----------------------")
    # snl_mrp.display_stationary_distribution()
    # print()
    # raise

    print("Reward Function")
    print("---------------")
    snl_mrp.display_reward_function()
    print()

    print("Value Function")
    print("--------------")
    snl_mrp.display_value_function(gamma=user_gamma)
    print()
