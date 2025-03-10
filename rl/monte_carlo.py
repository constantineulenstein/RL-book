'''Monte Carlo methods for working with Markov Reward Process and
Markov Decision Processes.

'''

from typing import Iterable, Iterator, TypeVar, Callable, Dict, Sequence, Mapping, Tuple
from rl.distribution import Categorical
from rl.approximate_dynamic_programming import (ValueFunctionApprox,
                                                QValueFunctionApprox,
                                                NTStateDistribution)
from rl.iterate import last
from rl.markov_decision_process import MarkovDecisionProcess, Policy, \
    TransitionStep, NonTerminal
from rl.policy import DeterministicPolicy, RandomPolicy, UniformPolicy
import rl.markov_process as mp
from rl.returns import returns
import itertools
import numpy as np
from collections import defaultdict

S = TypeVar('S')
A = TypeVar('A')


def mc_prediction(
    traces: Iterable[Iterable[mp.TransitionStep[S]]],
    approx_0: ValueFunctionApprox[S],
    γ: float,
    episode_length_tolerance: float = 1e-6
) -> Iterator[ValueFunctionApprox[S]]:
    '''Evaluate an MRP using the monte carlo method, simulating episodes
    of the given number of steps.

    Each value this function yields represents the approximated value
    function for the MRP after one additional epsiode.

    Arguments:
      traces -- an iterator of simulation traces from an MRP
      approx_0 -- initial approximation of value function
      γ -- discount rate (0 < γ ≤ 1), default: 1
      episode_length_tolerance -- stop iterating once γᵏ ≤ tolerance

    Returns an iterator with updates to the approximated value
    function after each episode.

    '''
    episodes: Iterator[Iterator[mp.ReturnStep[S]]] = \
        (returns(trace, γ, episode_length_tolerance) for trace in traces)
    f = approx_0
    yield f

    for episode in episodes:
        f = last(f.iterate_updates(
            [(step.state, step.return_)] for step in episode
        ))
        yield f


def mc_prediction_tabular(
    traces: Iterable[Iterable[mp.TransitionStep[S]]],
    γ: float,
    episode_length_tolerance: float = 1e-6
) -> Iterator[Mapping[S, float]]:

    episodes: Iterator[Iterator[mp.ReturnStep[S]]] = \
        (returns(trace, γ, episode_length_tolerance) for trace in traces)
    print(episodes)
    mapping = defaultdict(float)
    counter = defaultdict(int)
    for episode in episodes:
        for s in episode:
            state = s.state
            return_ = s.return_
            mapping[state] += 1/(counter[state]+1)*(return_ - mapping[state])
            counter[state] += 1
        yield mapping




    # sorted_returns_seq: Sequence[mp.ReturnStep[S]] = sorted(
    #     episodes,
    #     key=mp.ReturnStep[S].state.state
    # )
    # return {NonTerminal(s): np.mean([r.return_ for r in l])
    #         for s, l in itertools.groupby(
    #         sorted_returns_seq,
    #         key=mp.ReturnStep[S].state.state
    #     )}




def batch_mc_prediction(
    traces: Iterable[Iterable[mp.TransitionStep[S]]],
    approx: ValueFunctionApprox[S],
    γ: float,
    episode_length_tolerance: float = 1e-6,
    convergence_tolerance: float = 1e-5
) -> ValueFunctionApprox[S]:
    '''traces is a finite iterable'''
    return_steps: Iterable[mp.ReturnStep[S]] = \
        itertools.chain.from_iterable(
            returns(trace, γ, episode_length_tolerance) for trace in traces
        )
    return approx.solve(
        [(step.state, step.return_) for step in return_steps],
        convergence_tolerance
    )


def greedy_policy_from_qvf(
    q: QValueFunctionApprox[S, A],
    actions: Callable[[NonTerminal[S]], Iterable[A]]
) -> DeterministicPolicy[S, A]:
    '''Return the policy that takes the optimal action at each state based
    on the given approximation of the process's Q function.

    '''
    def optimal_action(s: S) -> A:
        _, a = q.argmax((NonTerminal(s), a) for a in actions(NonTerminal(s)))
        return a
    return DeterministicPolicy(optimal_action)


def epsilon_greedy_policy(
    q: QValueFunctionApprox[S, A],
    mdp: MarkovDecisionProcess[S, A],
    ε: float = 0.0
) -> Policy[S, A]:
    def explore(s: S, mdp=mdp) -> Iterable[A]:
        return mdp.actions(NonTerminal(s))
    return RandomPolicy(Categorical(
        {UniformPolicy(explore): ε,
         greedy_policy_from_qvf(q, mdp.actions): 1 - ε}
    ))


def glie_mc_control(
    mdp: MarkovDecisionProcess[S, A],
    states: NTStateDistribution[S],
    approx_0: QValueFunctionApprox[S, A],
    γ: float,
    ϵ_as_func_of_episodes: Callable[[int], float],
    episode_length_tolerance: float = 1e-6
) -> Iterator[QValueFunctionApprox[S, A]]:
    '''Evaluate an MRP using the monte carlo method, simulating episodes
    of the given number of steps.

    Each value this function yields represents the approximated value
    function for the MRP after one additional epsiode.

    Arguments:
      mdp -- the Markov Decision Process to evaluate
      states -- distribution of states to start episodes from
      approx_0 -- initial approximation of value function
      γ -- discount rate (0 ≤ γ ≤ 1)
      ϵ_as_func_of_episodes -- a function from the number of episodes
      to epsilon. epsilon is the fraction of the actions where we explore
      rather than following the optimal policy
      episode_length_tolerance -- stop iterating once γᵏ ≤ tolerance

    Returns an iterator with updates to the approximated Q function
    after each episode.

    '''
    q: QValueFunctionApprox[S, A] = approx_0
    p: Policy[S, A] = epsilon_greedy_policy(q, mdp, 1.0)
    yield q

    num_episodes: int = 0
    while True:
        trace: Iterable[TransitionStep[S, A]] = \
            mdp.simulate_actions(states, p)
        num_episodes += 1
        for step in returns(trace, γ, episode_length_tolerance):
            q = q.update([((step.state, step.action), step.return_)])
        p = epsilon_greedy_policy(q, mdp, ϵ_as_func_of_episodes(num_episodes))
        yield q


def greedy_policy_from_qvf_tabular(
    q: Mapping[S, Mapping[A, float]]
) -> DeterministicPolicy[S, A]:
    '''Return the policy that takes the optimal action at each state based
    on the given approximation of the process's Q function.

    '''

    def optimal_action(s: S) -> A:
        a = max(q[NonTerminal(s)], key=q[NonTerminal(s)].get)
        return a
    return DeterministicPolicy(optimal_action)

def epsilon_greedy_policy_tabular(
    q: Mapping[S, Mapping[A, float]],
    mdp: MarkovDecisionProcess[S, A],
    ε: float = 0.0
) -> Policy[S, A]:
    def explore(s: S, mdp=mdp) -> Iterable[A]:
        return mdp.actions(NonTerminal(s))

    return RandomPolicy(Categorical(
        {UniformPolicy(explore): ε,
         greedy_policy_from_qvf_tabular(q): 1 - ε}
    ))

def glie_mc_control_tabular(
    mdp: MarkovDecisionProcess[S, A],
    states: NTStateDistribution[S],
    γ: float,
    ϵ_as_func_of_episodes: Callable[[int], float],
    episode_length_tolerance: float = 1e-6
) -> Iterator[Mapping[S, Mapping[A, float]]]:
    #p: Policy[S, A] = epsilon_greedy_policy(q, mdp, 1.0)

    def explore(s: S, mdp=mdp) -> Iterable[A]:
        return mdp.actions(NonTerminal(s))

    p: Policy[S,A] = RandomPolicy(Categorical(
        {UniformPolicy(explore): 1}
    ))

    q = defaultdict(dict)
    counter = defaultdict(float)
    num_episodes: int = 0

    while True:
        trace: Iterable[TransitionStep[S, A]] = \
            mdp.simulate_actions(states, p)
        num_episodes += 1
        for step in returns(trace, γ, episode_length_tolerance):
            state = step.state
            if state not in q:
                q[state] = defaultdict(float)
            action = step.action
            return_ = step.return_
            q[state][action] += 1 / (counter[(state, action)] + 1) * (return_ - q[state][action])
            counter[(state, action)] += 1

        p = epsilon_greedy_policy_tabular(q, mdp, ϵ_as_func_of_episodes(num_episodes))
        yield q


