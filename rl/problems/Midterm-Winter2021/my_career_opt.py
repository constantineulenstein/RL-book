from dataclasses import dataclass
from typing import Tuple, Dict, Mapping, Iterable, Sequence
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.policy import FiniteDeterministicPolicy
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess
from rl.distribution import Categorical
from scipy.stats import poisson


@dataclass(frozen=True)
class CareerState:
    w: int


CareerOptMapping = Mapping[
    CareerState,
    Mapping[Tuple[int, int], Categorical[Tuple[CareerState, float]]]
]


class CareerOptMDP(FiniteMarkovDecisionProcess[CareerState, int]):

    def __init__(
            self,
            wage_cap: int,
            alpha: float,
            beta: float,
            hours: int
    ):
        self.wage_cap: int = wage_cap
        self.alpha: float = alpha
        self.beta: float = beta
        self.hours: int = hours

        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> CareerOptMapping:
        d: Dict[CareerState, Dict[Tuple[int, int], Categorical[Tuple[CareerState,
                                                                     float]]]] = {}
        for wage in range(1, self.wage_cap + 1):
            state: CareerState = CareerState(wage)

            # base_reward: float = - self.holding_cost * alpha
            d1: Dict[Tuple[int, int], Categorical[Tuple[CareerState, float]]] = {}

            for l in range(self.hours + 1):
                poisson_distr = poisson(self.alpha * l)
                for s in range(self.hours + 1 - l):
                    new_job_prob = self.beta * s / self.hours
                    reward = wage * (self.hours - l - s)
                    sr_probs_dict: Dict[Tuple[CareerState, float], float] = {}
                    same_prob : float = (1 - new_job_prob) * poisson_distr.pmf(0)
                    if wage == self.wage_cap:
                        sr_probs_dict[(state, reward)] = 1
                    elif wage == self.wage_cap - 1:
                        sr_probs_dict[(state, reward)] = same_prob
                        sr_probs_dict[(CareerState(self.wage_cap), reward)] = 1 - same_prob
                    else:
                        sr_probs_dict[(state, reward)] = same_prob
                        sr_probs_dict[(CareerState(wage + 1), reward)] = new_job_prob * poisson_distr.pmf(0) + poisson_distr.pmf(1)
                        for i in range(2, self.wage_cap - wage + 1):
                            if wage + i == self.wage_cap:
                                sr_probs_dict[(CareerState(self.wage_cap), reward)] = 1 - poisson_distr.cdf(i - 1)
                            else:
                                sr_probs_dict[(CareerState(wage + i), reward)] = poisson_distr.pmf(i)

                    d1[(l, s)] = Categorical(sr_probs_dict)

            d[state] = d1
        return d


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from pprint import pprint
    from rl.dynamic_programming import evaluate_mrp_result
    from rl.dynamic_programming import policy_iteration_result
    from rl.dynamic_programming import value_iteration_result

    hours: int = 10
    wage_cap: int = 30
    alpha: float = 0.08
    beta: float = 0.82
    gamma: float = 0.95

    co: CareerOptMDP = CareerOptMDP(
        hours=hours,
        wage_cap=wage_cap,
        alpha=alpha,
        beta=beta
    )

    _, opt_det_policy = value_iteration_result(co, gamma=gamma)
    pprint(opt_det_policy)
    wages: Iterable[int] = range(1, co.wage_cap + 1)
    opt_actions: Mapping[int, Tuple[int, int]] = {w: opt_det_policy.action_for[CareerState(w)] for w in wages}
    searching: Sequence[int] = [s for _, (_, s) in opt_actions.items()]
    learning: Sequence[int] = [l for _, (l, _) in opt_actions.items()]
    working: Sequence[int] = [co.hours - s - l for _, (l, s) in
                              opt_actions.items()]

    pprint(opt_actions)
    plt.xticks(wages)
    p1 = plt.bar(wages, searching, color='red')
    p2 = plt.bar(wages, learning, color='blue')
    p3 = plt.bar(wages, working, color='green')
    plt.legend((p1[0], p2[0], p3[0]), ('Job-Searching', 'Learning', 'Working'))
    plt.grid(axis='y')
    plt.xlabel("Hourly Wage Level")
    plt.ylabel("Hours Spent")
    plt.title("Career Optimization")
    plt.show()