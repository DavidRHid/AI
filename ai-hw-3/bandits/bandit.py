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


    def simulate(
        self
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        # TODO
        regret = np.zeros(self.N)
        return self.Qvalues, regret


    def choose_arm_egreedy(
        self
    ) -> int:
        # TODO
        return 0


    def choose_arm_edecay(
        self,
        t: int
    ) -> int:
        # TODO
        return 0


    def choose_arm_ucb(
        self,
        t: int
    ) -> int:
        # TODO
        return 0


    def pull_arm(
        self,
        idx: int,
    ) -> float:
        reward = 0
        for i in range(len(self.arms[idx]) - 1):
            mu_edge = self.edges[self.arms[idx][i]][self.arms[idx][i + 1]]["mu"]
            conditional_mean = np.log(mu_edge) - 0.5 * (self.conditional_sigma ** 2)
            reward -= np.exp(conditional_mean + self.conditional_sigma * np.random.randn())
        return reward


    def get_path_mean(
        self,
        idx: int,
    ) -> float:
        return -self.graph.all_path_means[idx]