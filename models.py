from typing import Protocol
from dataclasses import dataclass
from scipy.stats import entropy
import numpy as np
from indexers import Indexer, bottleneck_task_indexer
from tasks import option_choice_set_bottleneck


class Agent(Protocol):
    @staticmethod
    def _index(state: list, action: int = None, n_actions: int = None) -> tuple:
        ...

    def act(self, state, n_actions):
        ...

    def update_values(self) -> None:
        ...


def softargmax(x: np.ndarray, beta: float = 1) -> np.array:
    x = np.array(x)
    b = np.max(beta * x)
    y = np.exp(beta * x - b)
    return y / y.sum()


@dataclass
class DRA:
    q_size: tuple[int]
    _index: Indexer
    sigma_base: float = 1
    gamma: float = 0.98
    beta: float = 10
    lmda: float = 0.1
    learning_q: float = 0.1
    learning_sigma: float = 0.1
    n_trajectories: int = 10

    def __post_init__(self):
        self.q = np.zeros(self.q_size)
        self.sigma = self.sigma_base * np.ones(self.q_size)

    def act(self, state: list, n_actions: int):
        # index pointer to the q-table
        idx = self._index(state=state, n_actions=n_actions)

        # random draws from memory distribution
        zeta = np.random.randn(n_actions)
        prob_actions = softargmax(self.q[idx] + zeta * self.sigma[idx], self.beta)

        # choose action
        action = np.random.choice(np.arange(n_actions), p=prob_actions)

        return action, prob_actions, zeta

    def update_values(self, s, a, r, s1, prob_a) -> None:
        # define indices
        idx_sa = self._index(state=s, action=a)
        idx_s1 = self._index(state=s1, n_actions=len(prob_a))

        # compute prediction error and update values
        delta = r + self.gamma * np.max(self.q[idx_s1]) - self.q[idx_sa]
        self.q[idx_sa] += self.learning_q * delta

        return


@dataclass
class MaxEntRL:
    q_size: tuple[int]
    _index: Indexer
    alpha: float = 1
    gamma: float = 0.98
    learning_q: float = 0.1

    def __post_init__(self):
        self.q = np.zeros(self.q_size)
        if isinstance(self.q_size, int):
            self.v = np.zeros(self.q_size)
        else:
            self.v = np.zeros(self.q_size[:-1])

    def act(self, state: list, n_actions: int):
        # index pointer to the q-table
        idx_q = self._index(state=state, n_actions=n_actions)

        # probability of actions and chosen action
        prob_actions = softargmax(self.q[idx_q], 1 / self.alpha)
        action = np.random.choice(np.arange(n_actions), p=prob_actions)

        return action, prob_actions, None

    def update_values(self, s, a, r, s1, prob_a):
        # define indices
        idx_sa = self._index(state=s, action=a)
        idx_s = self._index(state=s)
        idx_q = self._index(state=s, n_actions=len(prob_a))
        idx_s1 = self._index(state=s1)

        # defining error term for value updates
        target = r + self.gamma * self.v[idx_s1]
        prediction = self.q[idx_sa]
        delta = target - prediction

        # value updates
        self.q[idx_sa] += self.learning_q * delta
        self.v[idx_s] = np.dot(prob_a, self.q[idx_q]) + self.alpha * entropy(prob_a)

        return


@dataclass
class MaxEntRL_bottleneck:
    q_size: tuple[int]
    _index: Indexer = bottleneck_task_indexer
    option_choice_set = staticmethod(option_choice_set_bottleneck)
    alpha: float = 1
    gamma: float = 0.98
    learning_q: float = 0.1

    def __post_init__(self):
        self.q = np.zeros((self.q_size))
        self.v = np.zeros(self.q_size)
        self.option_sets = [self.option_choice_set(i) for i in range(self.q_size - 1)]

    def all_common_indices(self, idx_sa):
        """Returns all indices with option in idx_sa"""
        option_id = self.option_sets[idx_sa]
        array_of_indices = np.array(np.where(self.option_sets == option_id)).transpose()
        return [tuple(idx) for idx in list(array_of_indices)]

    def act(self, state: list, n_actions: int):
        # index pointer to the q-table
        idx_q = self._index(state=state, n_actions=n_actions)

        # probability of actions and chosen action
        prob_actions = softargmax(self.q[idx_q], 1 / self.alpha)
        action = np.random.choice(np.arange(n_actions), p=prob_actions)

        return action, prob_actions, None

    def update_values(self, s, a, r, s1, prob_a):
        # define indices
        idx_sa = self._index(state=s, action=a)
        idx_s = self._index(state=s)
        idx_q = self._index(state=s, n_actions=len(prob_a))
        idx_s1 = self._index(state=s1)

        # value updates
        delta = r + self.gamma * self.v[idx_s1] - self.v[idx_s]
        self.q[idx_sa] += self.learning_q * delta
        self.v[idx_s] = np.dot(prob_a, self.q[idx_q]) + self.alpha * entropy(prob_a)

        return
