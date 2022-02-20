'''lambda-return and TD(lambda) methods for working with prediction and control

'''

from typing import Iterable, Iterator, TypeVar, List, Sequence, Mapping
from rl.function_approx import Gradient
import rl.markov_process as mp
from rl.markov_decision_process import NonTerminal
import numpy as np
from rl.approximate_dynamic_programming import ValueFunctionApprox
from rl.approximate_dynamic_programming import extended_vf
from rl.returns import returns
from collections import defaultdict

S = TypeVar('S')


def lambda_return_prediction(
        traces: Iterable[Iterable[mp.TransitionStep[S]]],
        approx_0: ValueFunctionApprox[S],
        γ: float,
        lambd: float
) -> Iterator[ValueFunctionApprox[S]]:
    '''Value Function Prediction using the lambda-return method given a
    sequence of traces.

    Each value this function yields represents the approximated value
    function for the MRP after an additional episode

    Arguments:
      traces -- a sequence of traces
      approx_0 -- initial approximation of value function
      γ -- discount rate (0 < γ ≤ 1)
      lambd -- lambda parameter (0 <= lambd <= 1)
    '''
    func_approx: ValueFunctionApprox[S] = approx_0
    yield func_approx

    for trace in traces:
        gp: List[float] = [1.]
        lp: List[float] = [1.]
        predictors: List[NonTerminal[S]] = []
        partials: List[List[float]] = []
        weights: List[List[float]] = []
        trace_seq: Sequence[mp.TransitionStep[S]] = list(trace)
        for t, tr in enumerate(trace_seq):
            for i, partial in enumerate(partials):
                partial.append(
                    partial[-1] +
                    gp[t - i] * (tr.reward - func_approx(tr.state)) +
                    (gp[t - i] * γ * extended_vf(func_approx, tr.next_state)
                     if t < len(trace_seq) - 1 else 0.)
                )
                weights[i].append(
                    weights[i][-1] * lambd if t < len(trace_seq)
                    else lp[t - i]
                )
            predictors.append(tr.state)
            partials.append([tr.reward +
                             (γ * extended_vf(func_approx, tr.next_state)
                              if t < len(trace_seq) - 1 else 0.)])
            weights.append([1. - (lambd if t < len(trace_seq) else 0.)])
            gp.append(gp[-1] * γ)
            lp.append(lp[-1] * lambd)
        responses: Sequence[float] = [np.dot(p, w) for p, w in
                                      zip(partials, weights)]
        for p, r in zip(predictors, responses):
            func_approx = func_approx.update([(p, r)])
        yield func_approx


def td_lambda_prediction(
        traces: Iterable[Iterable[mp.TransitionStep[S]]],
        approx_0: ValueFunctionApprox[S],
        γ: float,
        lambd: float
) -> Iterator[ValueFunctionApprox[S]]:
    '''Evaluate an MRP using TD(lambda) using the given sequence of traces.

    Each value this function yields represents the approximated value function
    for the MRP after an additional transition within each trace

    Arguments:
      transitions -- a sequence of transitions from an MRP which don't
                     have to be in order or from the same simulation
      approx_0 -- initial approximation of value function
      γ -- discount rate (0 < γ ≤ 1)
      lambd -- lambda parameter (0 <= lambd <= 1)
    '''
    func_approx: ValueFunctionApprox[S] = approx_0
    yield func_approx

    for trace in traces:
        el_tr: Gradient[ValueFunctionApprox[S]] = Gradient(func_approx).zero()
        for step in trace:
            x: NonTerminal[S] = step.state
            y: float = step.reward + γ * \
                       extended_vf(func_approx, step.next_state)
            el_tr = el_tr * (γ * lambd) + func_approx.objective_gradient(
                xy_vals_seq=[(x, y)],
                obj_deriv_out_fun=lambda x1, y1: np.ones(len(x1))
            )
            func_approx = func_approx.update_with_gradient(
                el_tr * (func_approx(x) - y)
            )
            yield func_approx


def td_lambda_tabular(
        traces: Iterable[Iterable[mp.TransitionStep[S]]],
        γ: float,
        lambd: float,
        episode_length_tolerance: float = 1e-6
) -> Iterator[Mapping[S, float]]:
    '''Evaluate an MRP using TD(lambda) using the given sequence of traces.

    Each value this function yields represents the approximated value function
    for the MRP after an additional transition within each trace

    Arguments:
      transitions -- a sequence of transitions from an MRP which don't
                     have to be in order or from the same simulation
      approx_0 -- initial approximation of value function
      γ -- discount rate (0 < γ ≤ 1)
      lambd -- lambda parameter (0 <= lambd <= 1)
    '''

    episodes: Iterator[Iterator[mp.ReturnStep[S]]] = \
        (returns(trace, γ, episode_length_tolerance) for trace in traces)

    mapping = defaultdict(float)
    counter = defaultdict(int)
    # el_tr = defaultdict(lambda: 1.0)

    for episode in episodes:
        temp_el_tr: float = 1.0
        temp_value_func: float = 0.0
        el_tr = {}
        for step in episode:
            state = step.state
            next_state = step.next_state
            reward = step.reward

            temp_el_tr *= γ * lambd

            keys = list(mapping.keys())
            for key in keys:
                if key in el_tr:
                    el_tr[key] *= γ * lambd
                else:
                    el_tr[key] = temp_el_tr
                if state == key:
                    el_tr[key] += 1

                mapping[key] += (1 / (counter[state] + 1)) * (reward + γ * mapping[next_state] - mapping[state]) * \
                                el_tr[key]

            temp_value_func += (1 / (counter[state] + 1)) * (reward + γ * mapping[next_state] - mapping[state]) * \
                               temp_el_tr
            if state not in mapping:
                mapping[state] = temp_value_func
            counter[state] += 1
            yield mapping

