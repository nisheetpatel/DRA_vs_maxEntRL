from typing import Protocol
from dataclasses import dataclass
from scipy.stats import entropy
import numpy as np
from tasks import option_choice_set_2afc


class Agent(Protocol):
    @staticmethod
    def _index(state: list, action: int = None, n_actions: int = None) -> tuple:
        ...

    def act(self, state, n_actions):
        ...

    def update_values(self) -> None:
        ...


def softargmax(x: np.array, beta: float = 1) -> np.array:
    x = np.array(x)
    b = np.max(beta * x)
    y = np.exp(beta * x - b)
    return y / y.sum()


def standard_indexer(state: list, action: int = None, n_actions: int = None) -> tuple:
    """Returns standard q-table index as tuple"""
    if (action is None) & (n_actions is None):
        return tuple(state)
    elif action is not None:
        return tuple(state + [action])
    else:    
        return tuple(state + [np.s_[:n_actions]])


def memory_2afc_task_indexer(state: list, action: int = None, n_actions: int = 2):
    """Returns q-table index for Memory_2AFC task as tuple"""
    if state == [-1]:
        return state[0]
        
    choice_set = option_choice_set_2afc(state=state[0])
    if action is None:
        return choice_set
    else:
        return choice_set[action]

@dataclass
class DRA:
    q_size: tuple[int]
    _index = staticmethod(standard_indexer)
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


    def act(self, state:list, n_actions: int):
        # index pointer to the q-table
        idx = self._index(state=state, n_actions=n_actions)

        # random draws from memory distribution
        zeta = np.random.randn(n_actions)
        prob_actions = softargmax(self.q[idx] + zeta * self.sigma[idx], self.beta)

        # chosen action
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
    _index = staticmethod(standard_indexer)
    alpha: float = 1
    gamma: float = 0.98
    learning_q: float = 0.1

    def __post_init__(self):
        self.q = np.zeros(self.q_size)
        if isinstance(self.q_size,int):
            self.v = np.zeros(self.q_size)
        else:
            self.v = np.zeros(self.q_size[:-1])

    def act(self, state:list, n_actions: int):
        # index pointer to the q-table
        idx_q = self._index(state=state, n_actions=n_actions)
        idx_v = self._index(state=state)
        
        value = self.v[idx_v]
        if not isinstance(self.v[idx_v], float):
            value = 0

        # defining advantage function
        advantage = self.q[idx_q] - value

        # probability of actions and chosen action
        prob_actions = softargmax(advantage, 1/self.alpha)
        action = np.random.choice(np.arange(n_actions), p=prob_actions)
        
        return action, prob_actions, None

    def update_values(self, s, a, r, s1, prob_a):
        # define indices
        idx_sa = self._index(state=s, action=a)
        idx_s  = self._index(state=s)
        idx_q  = self._index(state=s, n_actions=len(prob_a))
        idx_s1 = self._index(state=s1)

        # to resolve issues with value indexing for 2AFC task
        value = self.v[idx_s]
        if not isinstance(self.v[idx_s], float):
            value = np.dot(prob_a, self.v[idx_s])

        # value updates
        self.q[idx_sa] += self.learning_q * (r + self.gamma * self.v[idx_s1] - value)
        self.v[idx_s] = np.dot(prob_a, self.q[idx_q]) + self.alpha * entropy(prob_a)

        return


@dataclass
class MaxEntRL_2AFC:
    q_size: tuple[int]
    _index = staticmethod(standard_indexer)
    alpha: float = 1
    gamma: float = 0.98
    learning_q: float = 0.1

    def __post_init__(self):
        self.q = np.zeros((self.q_size,2))
        self.v = np.zeros(self.q_size)
        self.option_sets = np.array([option_choice_set_2afc(i) for i in range(36)])

    def all_common_indices(self, idx_sa):
        """Returns all indices with option in idx_sa"""
        option_id = self.option_sets[tuple(idx_sa)]
        array_of_indices = np.array(np.where(self.option_sets==option_id)).transpose()
        return [tuple(idx) for idx in list(array_of_indices)]

    def act(self, state:list, n_actions: int):
        # index pointer to the q-table
        idx_q = self._index(state=state, n_actions=n_actions)
        idx_v = self._index(state=state)
        
        # defining advantage function
        advantage = self.q[idx_q] - self.v[idx_v]

        # probability of actions and chosen action
        prob_actions = softargmax(advantage, 1/self.alpha)
        action = np.random.choice(np.arange(n_actions), p=prob_actions)
        
        return action, prob_actions, None

    def update_values(self, s, a, r, s1, prob_a):
        # define indices
        idx_sa = self._index(state=s, action=a)
        idx_s  = self._index(state=s)
        idx_q  = self._index(state=s, n_actions=len(prob_a))
        idx_s1 = self._index(state=s1)

        # value updates
        delta = r + self.gamma * self.v[idx_s1] - self.v[idx_s]
        for idx in self.all_common_indices(idx_sa=idx_sa):
            if list(idx) not in self.fixed_ids:
                self.q[idx] += self.learning_q * delta

        self.v[idx_s] = np.dot(prob_a, self.q[idx_q]) + self.alpha * entropy(prob_a)

        return