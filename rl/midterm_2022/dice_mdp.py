from typing import Tuple, Dict, Mapping
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.markov_process import FiniteMarkovRewardProcess, NonTerminal
from rl.dynamic_programming import value_iteration_result
from rl.distribution import Categorical
from collections import Counter
import itertools
import numpy as np


TuplePair = Tuple[Tuple, Tuple]

StateActionMapping = Mapping[
    TuplePair,
    Mapping[Tuple, Categorical[Tuple[TuplePair, int]]]
]


# def remove_subset(superset, subset):
#     superset = list(superset)
#     for elem in subset:
#         superset.pop(superset.index(elem))
#     return tuple(superset)

def extend_subset(superset, subset):
    superset = list(superset)
    subset = list(subset)
    superset.extend(subset)
    return tuple(sorted(superset))


class DiceMDP(FiniteMarkovDecisionProcess[int, TuplePair]):

    def __init__(
            self,
            n: int,
            k: int,
            c: int,
    ):
        self.n: int = n
        self.k: int = k
        self.c: int = c

        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> StateActionMapping:
        d: Dict[TuplePair, Dict[Tuple, Categorical[Tuple[TuplePair, int]]]] = {}
        sides = np.arange(1, self.k + 1)

        states = [(comb, (j)) for i in range(6)
                  for comb in list(itertools.combinations_with_replacement(sides, 6 - i))
                  for j in list(itertools.combinations_with_replacement(sides, i))]
        for state in states:
            table, hand = state[0], state[1]

            base_reward: int = 0

            actions_list = [np.unique(list(itertools.combinations(table, i)), axis=0)
                            for i in range(1, len(table) + 1)]

            d1: Dict[Tuple, Categorical[Tuple[TuplePair, int]]] = {}
            for actions in actions_list:
                for action in actions:
                    action = tuple(action)
                    sr_probs_dict: Dict[Tuple[TuplePair, int], float] = {}

                    new_hand = extend_subset(hand, action)
                    new_table_size = len(table) - len(action)

                    if new_table_size == 0:
                        one_count = Counter(new_hand)[1]
                        if one_count >= self.c:
                            sr_probs_dict[(((), new_hand), sum(new_hand))] = 1
                        else:
                            sr_probs_dict[(((), new_hand), base_reward)] = 1
                    else:
                        all_new_combs = list(itertools.product(sides, repeat=new_table_size))
                        counter = Counter(list(map(lambda x: tuple(sorted(x)), all_new_combs)))
                        for new_table in \
                                list(itertools.combinations_with_replacement(sides, new_table_size)):
                            sr_probs_dict[((new_table, new_hand), base_reward)] = \
                                counter[new_table]/len(all_new_combs)

                    d1[action] = Categorical(sr_probs_dict)

            d[state] = d1
        return d


if __name__ == '__main__':

    n = 6
    k = 4
    c = 1

    gamma = 1.0

    mdp: DiceMDP = DiceMDP(
        n=n,
        k=k,
        c=c
    )

    print("MDP Transition Map")
    print("------------------")
    print(mdp)
    _, opt_det_policy = value_iteration_result(mdp, gamma=gamma)
    print(opt_det_policy)

    implied_mrp: FiniteMarkovRewardProcess[TuplePair] = mdp.apply_finite_policy(opt_det_policy)

    print(implied_mrp)

    sides = np.arange(1, k + 1)
    all_starts = list(itertools.product(sides, repeat=n))
    counter = Counter(list(map(lambda x: tuple(sorted(x)), all_starts)))
    start_dict = {}
    for start_table in list(itertools.combinations_with_replacement(sides, n)):
        start_dict[NonTerminal((start_table, ()))] = counter[start_table] / len(all_starts)
    start_state_distribution = Categorical(start_dict)


    gen = implied_mrp.reward_traces(start_state_distribution)
    counter_tot = 0
    reward_tot = 0
    for trace in gen:
        if counter_tot == 1000000:
            break
        reward_tot += list(trace)[-1].reward
        counter_tot += 1

    print("Expected Reward:", reward_tot/counter_tot)