from typing import Protocol
import numpy as np
from dataclasses import dataclass
import gym.spaces


class Agent(Protocol):
    def act(self, state, action_space):
        ...


def softargmax(x: np.array, beta: float = 10) -> np.array:
    x = np.array(x)
    b = np.max(beta * x)
    y = np.exp(beta * x - b)
    return y / y.sum()


@dataclass
class DRA:
    q_size: tuple[int]
    alpha: float = 1

    def __post_init__(self):
        self.q_table = np.zeros(self.q_size)

    def q_idx(self, state: list, n_actions: int):
        return tuple(state + [np.s_[:n_actions]])

    def act(self, state:list, n_actions: int):
        prob_actions = softargmax(self.q_table[self.q_idx(state, n_actions)])
        return prob_actions

    def update_q_table(self):
        pass


@dataclass
class MaxEntRL:
    q_size: tuple[int]
    alpha: float = 1

    def __post_init__(self):
        self.q_table = np.zeros(self.q_size)

    def q_idx(self, state: list, n_actions: int):
        return tuple(state + [np.s_[:n_actions]])

    def act(self, state:list, n_actions: int):
        prob_actions = softargmax(self.q_table[self.q_idx(state, n_actions)])
        return prob_actions

    def update_q_table(self):
        pass