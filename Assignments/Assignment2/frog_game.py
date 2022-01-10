from dataclasses import dataclass
from typing import Optional, Mapping, Dict, Iterable
import numpy as np
import itertools
from rl.distribution import Categorical, Constant
from rl.markov_process import MarkovProcess, NonTerminal, State, Terminal, FiniteMarkovProcess
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class FrogState:
    position: int

class FrogGameFMP(FiniteMarkovProcess[FrogState]):
    def __init__(
            self,
            river_length: int = 10,
            max_frog_jump: int = 10
    ):
        if max_frog_jump > river_length:
            max_frog_jump = river_length
        self.river_length: int = river_length
        self.max_frog_jump: int = max_frog_jump
        super().__init__(self.get_transition_map())

    def get_transition_map(self) -> Mapping[FrogState, Categorical[FrogState]]:
        d: Dict[FrogState, Categorical[FrogState]] = {}

        for field in range(self.river_length):
            state_probs_map = {}

            for next_field in range(field+1, 11):
                state_probs_map[FrogState(next_field)] = 1 / (self.river_length-field)

            d[FrogState(field)] = Categorical(state_probs_map)
        return d

def get_traces(
        iterator: Iterable[Iterable[FrogState]],
        num_traces: int,
) -> np.ndarray:
    return np.array([[state for state in trace]
                     for trace in itertools.islice(iterator, num_traces)],dtype=object)

def plot_distribution(
        traces: np.ndarray,
        river_length: int
) -> float:
    counts: int = [len(trace) for trace in traces]
    plt.figure(figsize=(16,10))
    plt.hist(counts)
    plt.xlabel("Jumps")
    plt.ylabel("Count")
    plt.show()
    return np.mean(counts)





if __name__ == '__main__':

    frog_mp = FrogGameFMP(
        river_length=10,
        max_frog_jump=10
    )

    print("Transition Map")
    print("--------------")
    print(frog_mp)

    start = NonTerminal(FrogState(0))
    num_traces = 100000

    print("Traces")
    print("--------------")
    #print(snl_mp.traces(snl_mp.transition(state= self.SandLState(0))))

    traces = get_traces(frog_mp.traces(frog_mp.transition(start)), num_traces)
    print(plot_distribution(traces, river_length=10))