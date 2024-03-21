from typing import Tuple
from graph import Graph
import numpy as np
import numpy.typing as npt

class Bandit:
    def __init__(
        self,
        graph: Graph,
        conditional_sigma: float,
        strategy: int,
        value: float,
        N: int
    ):
        self.graph = graph
        self.arms = self.graph.arms
        self.edges = self.graph.edges
        self.conditional_sigma = conditional_sigma
        self.strategy = strategy
        self.value = value
        self.N = N
        self.Qvalues = np.zeros(len(self.arms))
        self.arm_counts = np.zeros(len(self.arms))

    def simulate(self) -> Tuple[npt.NDArray, npt.NDArray]:
        regret = np.zeros(self.N)
        if self.strategy == 0:
            self.choose_arm = self.choose_arm_egreedy
        elif self.strategy == 1:
            self.choose_arm = self.choose_arm_edecay
        else:
            self.choose_arm = self.choose_arm_ucb

        for t in range(self.N):
            idx = self.choose_arm(t)
            reward = self.pull_arm(idx)
            self.arm_counts[idx] += 1
            self.Qvalues[idx] += (reward - self.Qvalues[idx]) / self.arm_counts[idx]
            shortest_path = self.graph.shortest_path_ind()
            if shortest_path == idx:
                regret[t] = 0
            else:
                regret[t] = self.pull_arm(shortest_path) - reward
        return self.Qvalues, regret

    def choose_arm_egreedy(self, t: int) -> int:
        if np.random.rand() > self.value:
            return np.argmax(self.Qvalues)
        else:
            return np.random.randint(len(self.arms))

    def choose_arm_edecay(self, t: int) -> int:
        exploration_probs = np.minimum(1, (len(self.arms) * self.value) / (t + 1))
        if np.random.rand() > exploration_probs.min():
            return np.argmax(self.Qvalues)
        else:
            return np.random.randint(len(self.arms))
        
    def choose_arm_ucb(self, t: int) -> int:
        return np.argmax(self.Qvalues + self.value * np.sqrt(2*np.log(t + 1) / (self.arm_counts + 1)))
    
    def pull_arm(self, idx: int) -> float:
        reward = 0
        for i in range(len(self.arms[idx]) - 1):
            mu_edge = self.edges[self.arms[idx][i]][self.arms[idx][i + 1]]["mu"]
            conditional_mean = np.log(mu_edge) - 0.5 * (self.conditional_sigma ** 2)
            reward -= np.exp(conditional_mean + self.conditional_sigma * np.random.randn())
        return reward

    def get_path_mean(self, idx: int) -> float:
        return -self.graph.all_path_means[idx]