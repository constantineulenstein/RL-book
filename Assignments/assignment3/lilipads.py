from dataclasses import dataclass
from typing import Tuple, Dict, Mapping
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.policy import FiniteDeterministicPolicy
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess
from rl.distribution import Categorical
from scipy.stats import poisson
import itertools


@dataclass(frozen=True)
class FrogState:
    position: int


FrogStateActionMapping = Mapping[
    FrogState,
    Mapping[str, Categorical[Tuple[FrogState, float]]]
]


class LilipadMDP(FiniteMarkovDecisionProcess[FrogState, int]):

    def __init__(
            self,
            lilipads: int
    ):
        self.lilipads: int = lilipads
        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> FrogStateActionMapping:
        d: Dict[FrogState, Dict[str, Categorical[Tuple[FrogState, float]]]] = {}

        rewards: Dict[int, int] = {
            i: 0 for i in range(1, self.lilipads)
        }
        rewards[0] = -1
        rewards[self.lilipads] = 1

        for i in range(1, self.lilipads):

            state: FrogState = FrogState(i)
            d1: Dict[str, Categorical[Tuple[FrogState, float]]] = {}
            # first action A
            sr_probs_dict_action_a: Dict[Tuple[FrogState, float], float] = \
                {
                    (FrogState(i - 1), rewards[i - 1]): i / self.lilipads,
                    (FrogState(i + 1), rewards[i + 1]): (self.lilipads - i) / self.lilipads,
                }
            # then action B
            sr_probs_dict_action_b: Dict[Tuple[FrogState, float], float] = \
                {(FrogState(j), rewards[j]):
                     1 / self.lilipads for j in range(self.lilipads + 1) if j != i}

            d1["A"] = Categorical(sr_probs_dict_action_a)
            d1["B"] = Categorical(sr_probs_dict_action_b)
            d[state] = d1
        return d


if __name__ == '__main__':
    from pprint import pprint

    lilipads = 9
    user_gamma = 0.9

    lilipad_mdp: FiniteMarkovDecisionProcess[FrogState, int] = \
        LilipadMDP(
            lilipads=lilipads,
        )

    print("MDP Transition Map")
    print("------------------")
    print(lilipad_mdp)

    potential_actions = list(itertools.product("AB", repeat=lilipads-1))

    best_value_func = 0
    for actions in potential_actions:
        fdp: FiniteDeterministicPolicy[FrogState, int] = \
            FiniteDeterministicPolicy(
                {FrogState(i+1): action for i, action in enumerate(actions)}
            )


        print("Deterministic Policy Map")
        print("------------------------")
        print(fdp)


        implied_mrp: FiniteMarkovRewardProcess[FrogState] = \
            lilipad_mdp.apply_finite_policy(fdp)


        print("Implied MRP Transition Reward Map")
        print("---------------------")
        print(implied_mrp)

        value_func = implied_mrp.get_value_function_vec(user_gamma)

        print("Value function")
        print("---------------------")
        print(value_func)

        if sum(value_func) > best_value_func:
            best_value_func = sum(value_func)
            optimal_value_func = value_func
            best_fdp = fdp


    print("Optimal Policy Map")
    print("------------------------")
    print(best_fdp)

    print("Optimal Value function")
    print("---------------------")
    print(optimal_value_func)






