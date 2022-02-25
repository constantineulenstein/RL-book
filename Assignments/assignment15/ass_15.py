from typing import Sequence, Tuple, Mapping
from collections import defaultdict
from random import sample
import numpy as np

S = str
DataType = Sequence[Sequence[Tuple[S, float]]]
ProbFunc = Mapping[S, Mapping[S, float]]
RewardFunc = Mapping[S, float]
ValueFunc = Mapping[S, float]


def get_state_return_samples(
        data: DataType
) -> Sequence[Tuple[S, float]]:
    """
    prepare sequence of (state, return) pairs.
    Note: (state, return) pairs is not same as (state, reward) pairs.
    """
    return [(s, sum(r for (_, r) in l[i:]))
            for l in data for i, (s, _) in enumerate(l)]


def get_mc_value_function(
        state_return_samples: Sequence[Tuple[S, float]]
) -> ValueFunc:
    """
    Implement tabular MC Value Function compatible with the interface defined above.
    """
    vfc = defaultdict(float)
    counter = defaultdict(int)
    for (state, return_) in state_return_samples:
        vfc[state] += 1 / (counter[state] + 1) * (return_ - vfc[state])
        counter[state] += 1

    return vfc


def get_state_reward_next_state_samples(
        data: DataType
) -> Sequence[Tuple[S, float, S]]:
    """
    prepare sequence of (state, reward, next_state) triples.
    """
    return [(s, r, l[i + 1][0] if i < len(l) - 1 else 'T')
            for l in data for i, (s, r) in enumerate(l)]


def get_probability_and_reward_functions(
        srs_samples: Sequence[Tuple[S, float, S]]
) -> Tuple[ProbFunc, RewardFunc]:
    """
    Implement code that produces the probability transitions and the
    reward function compatible with the interface defined above.
    """
    # ProbFunc = Mapping[S, Mapping[S, float]]
    # RewardFunc = Mapping[S, float]
    probfunc = defaultdict(lambda: defaultdict(float))
    rewardfunc = defaultdict(float)
    counter = defaultdict(int)
    for (state, reward, next_state) in srs_samples:
        probfunc[state][next_state] += 1
        rewardfunc[state] += reward
        counter[state] += 1
    probfunc = {state: {next_state: probfunc[state][next_state] / counter[state] for next_state in probfunc[state]} for
                state in probfunc}
    rewardfunc = {state: rewardfunc[state] / counter[state] for state in rewardfunc}
    return probfunc, rewardfunc


def get_mrp_value_function(
        prob_func: ProbFunc,
        reward_func: RewardFunc
) -> ValueFunc:
    """
    Implement code that calculates the MRP Value Function from the probability
    transitions and reward function, compatible with the interface defined above.
    Hint: Use the MRP Bellman Equation and simple linear algebra
    """
    sz = len(prob_func.keys())
    mat = np.zeros((sz, sz))

    for i, state in enumerate(sorted(prob_func)):
        for j, next_state in enumerate(sorted(prob_func)):
            mat[i,j] = prob_func[state][next_state]
    vec = list(reward_func.values())
    vfc_values = np.linalg.solve(np.eye(sz) - mat, vec)
    vfc = {state: vfc_values[i] for i, state in enumerate(sorted(prob_func))}
    return vfc




def get_td_value_function(
        srs_samples: Sequence[Tuple[S, float, S]],
        num_updates: int = 300000,
        learning_rate: float = 0.3,
        learning_rate_decay: int = 30
) -> ValueFunc:
    """
    Implement tabular TD(0) (with experience replay) Value Function compatible
    with the interface defined above. Let the step size (alpha) be:
    learning_rate * (updates / learning_rate_decay + 1) ** -0.5
    so that Robbins-Monro condition is satisfied for the sequence of step sizes.
    """
    vfc = defaultdict(float)
    for update in range(1, num_updates + 1):
        (state, reward, next_state) = sample(srs_samples, 1)[0]
        vfc[state] += learning_rate * (update / learning_rate_decay + 1) ** -0.5 * (
                    reward + vfc[next_state] - vfc[state])
    return vfc


def get_lstd_value_function(
        srs_samples: Sequence[Tuple[S, float, S]]
) -> ValueFunc:
    """
    Implement LSTD Value Function compatible with the interface defined above.
    Hint: Tabular is a special case of linear function approx where each feature
    is an indicator variables for a corresponding state and each parameter is
    the value function for the corresponding state.
    """


if __name__ == '__main__':
    given_data: DataType = [
        [('A', 2.), ('A', 6.), ('B', 1.), ('B', 2.)],
        [('A', 3.), ('B', 2.), ('A', 4.), ('B', 2.), ('B', 0.)],
        [('B', 3.), ('B', 6.), ('A', 1.), ('B', 1.)],
        [('A', 0.), ('B', 2.), ('A', 4.), ('B', 4.), ('B', 2.), ('B', 3.)],
        [('B', 8.), ('B', 2.)]
    ]

    sr_samps = get_state_return_samples(given_data)

    print("------------- MONTE CARLO VALUE FUNCTION --------------")
    print(get_mc_value_function(sr_samps))

    srs_samps = get_state_reward_next_state_samples(given_data)
    # print(srs_samps)

    pfunc, rfunc = get_probability_and_reward_functions(srs_samps)
    # print(pfunc)
    # print(rfunc)
    print("-------------- MRP VALUE FUNCTION ----------")
    print(get_mrp_value_function(pfunc, rfunc))

    print("------------- TD VALUE FUNCTION --------------")
    print(get_td_value_function(srs_samps))

    print("------------- LSTD VALUE FUNCTION --------------")
    print(get_lstd_value_function(srs_samps))
