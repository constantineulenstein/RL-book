from dataclasses import dataclass
from typing import Tuple, Dict, Mapping
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.policy import FiniteDeterministicPolicy
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess
from rl.distribution import Categorical
from scipy.stats import poisson


@dataclass(frozen=True)
class InventoryState:
    # Stores A and B
    on_hand_a: int
    on_order_a: int
    on_hand_b: int
    on_order_b: int

    def inventory_position(self) -> Tuple[int, int]:
        return self.on_hand_a + self.on_order_a, self.on_hand_b + self.on_order_b


# Tuple[int,int, int]: first int is supply orders for store A, second supply orders for store B, third
# transfers from A to B
InvOrderMapping = Mapping[
    InventoryState,
    Mapping[Tuple[int, int, int], Categorical[Tuple[InventoryState, float]]]
]


class TwoStoreInventoryMDPCap(FiniteMarkovDecisionProcess[InventoryState, int]):

    def __init__(
            self,
            capacity_a: int,
            poisson_lambda_a: float,
            holding_cost_a: float,
            stockout_cost_a: float,
            capacity_b: int,
            poisson_lambda_b: float,
            holding_cost_b: float,
            stockout_cost_b: float,
            transfer_cost_supply: float,
            transfer_cost_stores: float
    ):
        self.capacity_a: int = capacity_a
        self.poisson_lambda_a: float = poisson_lambda_a
        self.holding_cost_a: float = holding_cost_a
        self.stockout_cost_a: float = stockout_cost_a
        self.capacity_b: int = capacity_b
        self.poisson_lambda_b: float = poisson_lambda_b
        self.holding_cost_b: float = holding_cost_b
        self.stockout_cost_b: float = stockout_cost_b

        self.poisson_distr_a = poisson(poisson_lambda_a)
        self.poisson_distr_b = poisson(poisson_lambda_b)

        self.transfer_cost_supply: float = transfer_cost_supply
        self.transfer_cost_stores: float = transfer_cost_stores

        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> InvOrderMapping:
        d: Dict[InventoryState, Dict[Tuple[int, int, int], Categorical[Tuple[InventoryState,
                                                                             float]]]] = {}

        for alpha_a in range(self.capacity_a + 1):
            for beta_a in range(self.capacity_a + 1 - alpha_a):
                for alpha_b in range(self.capacity_b + 1):
                    for beta_b in range(self.capacity_b + 1 - alpha_b):
                        state: InventoryState = InventoryState(alpha_a, beta_a, alpha_b, beta_b)
                        ip_a, ip_b = state.inventory_position()

                        d1: Dict[Tuple[int, int, int], Categorical[Tuple[InventoryState, float]]] = {}

                        for order_a in range(self.capacity_a - ip_a + 1):
                            for order_b in range(self.capacity_b - ip_b + 1):
                                # max inventory transfer from B to A: min(alpha_b, capacity_a-alpha_a-beta_a)
                                # max inventory transfer from A to B: min(alpha_a, capacity_b-alpha_b-beta_b)
                                for inventory_transfer in range(-min(alpha_b, self.capacity_a - ip_a),
                                                                min(alpha_a, self.capacity_b - ip_b) + 1):
                                    sr_probs_dict: Dict[Tuple[InventoryState, float], float] = {}
                                    base_reward_a: float = - self.holding_cost_a * alpha_a
                                    base_reward_b: float = - self.holding_cost_b * alpha_b
                                    if order_a > 0:
                                        base_reward_a -=  self.transfer_cost_supply
                                    if order_b > 0:
                                        base_reward_b -= self.transfer_cost_supply

                                    # probablity/reward that A and/or B ran out of inventory
                                    prob_a_out_of_inv: float = 1 - self.poisson_distr_a.cdf(ip_a - 1)
                                    prob_b_out_of_inv: float = 1 - self.poisson_distr_b.cdf(ip_b - 1)

                                    reward_a: float = base_reward_a - self.stockout_cost_a * \
                                                      (prob_a_out_of_inv * (self.poisson_lambda_a - ip_a) +
                                                       ip_a * self.poisson_distr_a.pmf(ip_a))

                                    reward_b: float = base_reward_b - self.stockout_cost_b * \
                                                      (prob_b_out_of_inv * (self.poisson_lambda_b - ip_b) +
                                                       ip_b * self.poisson_distr_b.pmf(ip_b))

                                    if inventory_transfer > 0:
                                        base_reward_a -= self.transfer_cost_stores / 2
                                        base_reward_b -= self.transfer_cost_stores / 2
                                        reward_a -= self.transfer_cost_stores / 2
                                        reward_b -= self.transfer_cost_stores / 2

                                    # If none are out of stock
                                    for i in range(ip_a - inventory_transfer):
                                        for j in range(ip_b + inventory_transfer):
                                            sr_probs_dict[(InventoryState(ip_a - inventory_transfer - i, order_a,
                                                                          ip_b + inventory_transfer - j, order_b),
                                                           base_reward_a + base_reward_b)] = \
                                                self.poisson_distr_a.pmf(i) * self.poisson_distr_b.pmf(j)
                                            # Note that probs are independent
                                    # If A is out of stock
                                    for k in range(ip_b + inventory_transfer):
                                        sr_probs_dict[(InventoryState(0, order_a, ip_b + inventory_transfer - k, order_b),
                                                       reward_a + base_reward_b)] = \
                                            prob_a_out_of_inv * self.poisson_distr_b.pmf(k)
                                    # If B is out of stock
                                    for l in range(ip_a - inventory_transfer):
                                        sr_probs_dict[(InventoryState(ip_a - inventory_transfer - l, order_a, 0, order_b),
                                                       base_reward_a + reward_b)] = \
                                            self.poisson_distr_a.pmf(l) * prob_b_out_of_inv
                                    # If A and B are out of stock
                                    sr_probs_dict[(InventoryState(0, order_a, 0, order_b),
                                                   reward_a + reward_b)] = \
                                        prob_a_out_of_inv * prob_b_out_of_inv

                                    d1[(order_a, order_b, inventory_transfer)] = Categorical(sr_probs_dict)
                        d[state] = d1
        return d


if __name__ == '__main__':
    from pprint import pprint

    user_capacity_a = 4
    user_poisson_lambda_a = 1.0
    user_holding_cost_a = 1.0
    user_stockout_cost_a = 20.0

    user_capacity_b = 4
    user_poisson_lambda_b = 2.0
    user_holding_cost_b = 1.0
    user_stockout_cost_b = 10.0

    transfer_cost_stores = 2
    transfer_cost_supply = 1

    user_gamma = 0.9

    tsi_mdp: FiniteMarkovDecisionProcess[InventoryState, int] = \
        TwoStoreInventoryMDPCap(
            capacity_a=user_capacity_a,
            poisson_lambda_a=user_poisson_lambda_a,
            holding_cost_a=user_holding_cost_a,
            stockout_cost_a=user_stockout_cost_a,
            capacity_b=user_capacity_b,
            poisson_lambda_b=user_poisson_lambda_b,
            holding_cost_b=user_holding_cost_b,
            stockout_cost_b=user_stockout_cost_b,
            transfer_cost_stores=transfer_cost_stores,
            transfer_cost_supply=transfer_cost_supply
        )

    print("MDP Transition Map")
    print("------------------")
    print(tsi_mdp)


    # fdp: FiniteDeterministicPolicy[InventoryState, int] = \
    #     FiniteDeterministicPolicy(
    #         {InventoryState(alpha, beta): user_capacity - (alpha + beta)
    #          for alpha in range(user_capacity + 1)
    #          for beta in range(user_capacity + 1 - alpha)}
    #     )
    #
    # print("Deterministic Policy Map")
    # print("------------------------")
    # print(fdp)
    #
    # implied_mrp: FiniteMarkovRewardProcess[InventoryState] = \
    #     si_mdp.apply_finite_policy(fdp)
    # print("Implied MP Transition Map")
    # print("--------------")
    # print(FiniteMarkovProcess(
    #     {s.state: Categorical({s1.state: p for s1, p in v.table().items()})
    #      for s, v in implied_mrp.transition_map.items()}
    # ))
    #
    # print("Implied MRP Transition Reward Map")
    # print("---------------------")
    # print(implied_mrp)
    #
    # print("Implied MP Stationary Distribution")
    # print("-----------------------")
    # implied_mrp.display_stationary_distribution()
    # print()
    #
    # print("Implied MRP Reward Function")
    # print("---------------")
    # implied_mrp.display_reward_function()
    # print()
    #
    # print("Implied MRP Value Function")
    # print("--------------")
    # implied_mrp.display_value_function(gamma=user_gamma)
    # print()

    from rl.dynamic_programming import evaluate_mrp_result
    from rl.dynamic_programming import policy_iteration_result
    from rl.dynamic_programming import value_iteration_result

    # print("Implied MRP Policy Evaluation Value Function")
    # print("--------------")
    # pprint(evaluate_mrp_result(implied_mrp, gamma=user_gamma))
    # print()

    print("MDP Policy Iteration Optimal Value Function and Optimal Policy")
    print("--------------")
    opt_vf_pi, opt_policy_pi = policy_iteration_result(
        tsi_mdp,
        gamma=user_gamma
    )
    pprint(opt_vf_pi)
    print(opt_policy_pi)
    print()

    # print("MDP Value Iteration Optimal Value Function and Optimal Policy")
    # print("--------------")
    # opt_vf_vi, opt_policy_vi = value_iteration_result(tsi_mdp, gamma=user_gamma)
    # pprint(opt_vf_vi)
    # print(opt_policy_vi)
    # print()
