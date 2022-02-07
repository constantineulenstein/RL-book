from dataclasses import dataclass
from typing import Tuple, Dict, Mapping
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.policy import FiniteDeterministicPolicy
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess
from rl.distribution import Categorical
from scipy.stats import poisson


@dataclass(frozen=True)
class GridState:
    x: int
    y: int


StateActionMapping = Mapping[
    GridState,
    Mapping[str, Categorical[Tuple[GridState, int]]]
]


class GridMazeMDP(FiniteMarkovDecisionProcess[GridState, int]):

    def __init__(
            self,
            grid_maze: dict,
    ):
        self.grid_maze: dict = {key: value for (key, value) in grid_maze.items() if
                                (value == "SPACE" or value == "GOAL")}

        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> StateActionMapping:
        d: Dict[GridState, Dict[str, Categorical[Tuple[GridState, int]]]] = {}
        action_dict = {"L": (-1, 0), "R": (1, 0), "U": (0, 1), "D": (0, -1)}
        for (x, y) in self.grid_maze.keys():
            if self.grid_maze[(x, y)] == "GOAL":
                continue
            state: GridState = GridState(x, y)
            reward: int = -1
            d1: Dict[str, Categorical[Tuple[GridState, int]]] = {}

            for action in ["L", "R", "U", "D"]:
                sr_probs_dict: Dict[Tuple[GridState, int], int] = {}

                new_x = x + action_dict[action][0]
                new_y = y + action_dict[action][1]
                if (new_x, new_y) in self.grid_maze.keys():
                    sr_probs_dict[(GridState(new_x, new_y), reward)] = 1
                    d1[action] = Categorical(sr_probs_dict)

            d[state] = d1

        return d


if __name__ == '__main__':
    from pprint import pprint

    SPACE = 'SPACE'
    BLOCK = 'BLOCK'
    GOAL = 'GOAL'

    maze_grid = {(0, 0): SPACE, (0, 1): BLOCK, (0, 2): SPACE, (0, 3): SPACE, (0, 4): SPACE,
                 (0, 5): SPACE, (0, 6): SPACE, (0, 7): SPACE, (1, 0): SPACE, (1, 1): BLOCK,
                 (1, 2): BLOCK, (1, 3): SPACE, (1, 4): BLOCK, (1, 5): BLOCK, (1, 6): BLOCK,
                 (1, 7): BLOCK, (2, 0): SPACE, (2, 1): BLOCK, (2, 2): SPACE, (2, 3): SPACE,
                 (2, 4): SPACE, (2, 5): SPACE, (2, 6): BLOCK, (2, 7): SPACE, (3, 0): SPACE,
                 (3, 1): SPACE, (3, 2): SPACE, (3, 3): BLOCK, (3, 4): BLOCK, (3, 5): SPACE,
                 (3, 6): BLOCK, (3, 7): SPACE, (4, 0): SPACE, (4, 1): BLOCK, (4, 2): SPACE,
                 (4, 3): BLOCK, (4, 4): SPACE, (4, 5): SPACE, (4, 6): SPACE, (4, 7): SPACE,
                 (5, 0): BLOCK, (5, 1): BLOCK, (5, 2): SPACE, (5, 3): BLOCK, (5, 4): SPACE,
                 (5, 5): BLOCK, (5, 6): SPACE, (5, 7): BLOCK, (6, 0): SPACE, (6, 1): BLOCK,
                 (6, 2): BLOCK, (6, 3): BLOCK, (6, 4): SPACE, (6, 5): BLOCK, (6, 6): SPACE,
                 (6, 7): SPACE, (7, 0): SPACE, (7, 1): SPACE, (7, 2): SPACE, (7, 3): SPACE,
                 (7, 4): SPACE, (7, 5): BLOCK, (7, 6): BLOCK, (7, 7): GOAL}

    user_gamma = 1

    maze_mdp: FiniteMarkovDecisionProcess[GridState, int] = \
        GridMazeMDP(
            grid_maze=maze_grid,
        )

    print("MDP Transition Map")
    print("------------------")
    print(maze_mdp)

    from rl.dynamic_programming import value_iteration_result

    print("MDP Value Iteration Optimal Value Function and Optimal Policy")
    print("--------------")
    opt_vf_vi, opt_policy_vi = value_iteration_result(maze_mdp, gamma=user_gamma)
    pprint(opt_vf_vi)
    print(opt_policy_vi)
    print()
