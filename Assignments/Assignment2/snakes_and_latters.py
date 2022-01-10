from dataclasses import dataclass
from typing import Optional, Mapping, Dict, Iterable
import numpy as np
import itertools
from rl.distribution import Categorical, Constant
from rl.markov_process import MarkovProcess, NonTerminal, State, Terminal, FiniteMarkovProcess
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SandLState(State):
    position: int


class SnakeAndLaddersFMP(FiniteMarkovProcess[SandLState]):
    def __init__(
            self,
            num_fields: int = 100,
            extra_steps: Dict = None
    ):
        self.num_fields: int = num_fields
        self.extra_steps: Dict = extra_steps

        super().__init__(self.get_transition_map())

    def get_transition_map(self) -> Mapping[SandLState, Categorical[SandLState]]:
        d: Dict[SandLState, Categorical[SandLState]] = {}

        for field in range(self.num_fields):
            state_probs_map = {}

            for i in range(1, 7):
                next_state = field + i

                if i + field in list(self.extra_steps.keys()):
                    next_state = self.extra_steps[field + i]
                state_probs_map[SandLState(next_state)] = 1 / 6

            d[SandLState(field)] = Categorical(state_probs_map)
        return d


def get_traces(
        iterator: Iterable[Iterable[SandLState]],
        num_traces: int,
) -> np.ndarray:
    return np.array([[state for state in trace]
                     for trace in itertools.islice(iterator, num_traces)],dtype=object)

def plot_distribution(
        traces: np.ndarray
) -> None:
    counts: int = [len(trace) for trace in traces]
    plt.figure(figsize=(16,10))
    plt.hist(counts, bins=50)
    plt.xlabel("Dice Rolls")
    plt.ylabel("Count")
    plt.show()
    return

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
    snl_mp = SnakeAndLaddersFMP(
        num_fields=100,
        extra_steps=snl_steps
    )

    print("Transition Map")
    print("--------------")
    print(snl_mp)

    start = NonTerminal(SandLState(0))
    num_traces = 10000

    print("Traces")
    print("--------------")
    #print(snl_mp.traces(snl_mp.transition(state= self.SandLState(0))))

    traces = get_traces(snl_mp.traces(snl_mp.transition(start)), num_traces)
    plot_distribution(traces)