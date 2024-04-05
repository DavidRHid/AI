import numpy as np
import numpy.typing as npt


class Gridworld_HMM:
    def __init__(self, size, epsilon: float = 0, walls: list = []):
        if walls:
            self.grid = np.zeros(size)
            for cell in walls:
                self.grid[cell] = 1
        else:
            self.grid = np.random.randint(2, size=size)

        self.init = ((1 - self.grid) / np.sum(self.grid)).flatten('F')

        self.epsilon = epsilon
        self.trans = self.initT()
        self.obs = self.initO()

    def neighbors(self, cell):
        i, j = cell
        M, N = self.grid.shape
        adjacent = [(i, j), (i, j + 1), (i + 1, j), (i, j - 1), (i - 1, j)]
        neighbors = []
        for a1, a2 in adjacent:
            if 0 <= a1 < M and 0 <= a2 < N and self.grid[a1, a2] == 0:
                neighbors.append((a1, a2))
        return neighbors


    """
    4.1 and 4.2. Transition and observation probabilities
    """

    def initT(self):
        """
        Create and return nxn transition matrix, where n = size of grid.
        """
        T = np.zeros((self.grid.size, self.grid.size))
        for i in range(self.grid.size):
            neighbors = self.neighbors((i // self.grid.shape[1], i % self.grid.shape[1]))
            for neighbor in neighbors:
                j = neighbor[0] * self.grid.shape[1] + neighbor[1]
                T[i, j] = 1 / len(neighbors)
        return T

    def initO(self):
        """
        Create and return 16xn matrix of observation probabilities, where n = size of grid.
        """
        O = np.zeros((16, self.grid.size))
        for i in range(self.grid.size):
            neighbors = self.neighbors((i // self.grid.shape[1], i % self.grid.shape[1]))
            north = 0 if (i // self.grid.shape[1] - 1, i % self.grid.shape[1]) in neighbors else 1
            east = 0 if (i // self.grid.shape[1], i % self.grid.shape[1] + 1) in neighbors else 1
            south = 0 if (i // self.grid.shape[1] + 1, i % self.grid.shape[1]) in neighbors else 1
            west = 0 if (i // self.grid.shape[1], i % self.grid.shape[1] - 1) in neighbors else 1
            trueval = 8 * north + 4 * east + 2 * south + west
            for j in range(16):
                O[j, i] = ((1 - self.epsilon) ** (4 - (bin(trueval^j).count('1')))) * (self.epsilon ** (bin(trueval^j).count('1')))
        return O


    """
    4.3 Inference: Forward, backward, filtering, smoothing
    """

    def forward(self, alpha: npt.ArrayLike, observation: int):
        """Perform one iteration of the forward algorithm.
        Args:
          alpha (np.ndarray): Current belief state.
          observation (int): Integer representation of bitstring observation.
        Returns:
          np.ndarray: Updated belief state.
        """
        return alpha @ self.trans * self.obs[observation]


    def backward(self, beta: npt.ArrayLike, observation: int):
        """Perform one iteration of the backward algorithm.
        Args:
          beta (np.ndarray): Current array of probabilities.
          observation (int): Integer representation of bitstring observation.
        Returns:
          np.ndarray: Updated array.
        """

        return beta * self.obs[observation] @ self.trans.T


    def filtering(self, observations: list[int]):
        """Perform filtering over all observations.
        Args:
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Alpha vectors at each timestep.
          np.ndarray: Estimated belief state at each timestep.
        """
        alphas = np.zeros((len(observations), self.grid.size))
        belief = np.zeros((len(observations), self.grid.size))
        for i, observation in enumerate(observations):
            if i == 0:
                alphas[i] = self.init
            else:
                alphas[i] = self.forward(alphas[i - 1], observation)
            belief[i] = alphas[i] / np.sum(alphas[i])
        return alphas, belief


    def smoothing(self, observations: list[int]):
        """Perform smoothing over all observations.
        Args:
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Beta vectors at each timestep.
          np.ndarray: Smoothed belief state at each timestep.
        """
        alphas, belief = self.filtering(observations)
        betas = np.zeros_like(alphas)
        for i in range(len(observations) - 1, -1, -1):
            if i == len(observations) - 1:
                betas[i] = np.ones_like(self.init)
            else:
                betas[i] = self.backward(betas[i + 1], observations[i + 1])
            belief[i] = alphas[i] * betas[i] / np.sum(alphas[i] * betas[i])
        return betas, belief


    """
    4.4. Parameter learning: Baum-Welch
    """

    def baum_welch(self, observations: list[int]):
        """Learn observation probabilities using the Baum-Welch algorithm.
        Updates self.obs in place.
        Args:
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Learned 16xn matrix of observation probabilities, where n = size of grid.
          list[float]: List of data likelihoods at each iteration.
        """
        data_likelihoods = []
        for _ in range(100):
            alphas, _ = self.filtering(observations)
            betas, _ = self.smoothing(observations)
            xi = np.zeros((len(observations), self.grid.size, self.grid.size))
            for i in range(len(observations) - 1):
                xi[i] = np.outer(alphas[i], betas[i + 1])

            for j in range(self.grid.size):
                for k in range(self.grid.size):
                    self.obs[:, j] = np.sum(xi[:, j, k] * (observations == k)) / np.sum(xi[:, j, k])

            data_likelihoods.append(np.sum(np.log(np.sum(alphas, axis=1))))
        return self.obs, data_likelihoods