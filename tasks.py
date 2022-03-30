from ctypes import Union
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple
import gym.spaces
import numpy as np


class Environment(Protocol):
    @property
    def action_space() -> gym.spaces.Discrete:
        ...

    def reset() -> list[int]:
        ...

    def step() -> Tuple[list, int, bool, dict]:
        ...


@dataclass
class EnvironmentAttributes:
    n_states: int
    transitions: Optional[list[int, int]]
    rewards: Optional[list]

    @property
    def transition_matrix(self):
        try:
            transition_matrix = np.zeros(shape=(self.n_states, self.n_states))
            for state, next_state in self.transitions:
                transition_matrix[state, next_state] = 1
        except NameError:
            raise "Cannot create transition matrix without user-defined transitions"
        return transition_matrix


class ZiebartTask:
    """
    Task adapted from the last figure of Ziebart et al. 2008.
    This has has three possible trajectories, all of which yield the same reward.
    """

    def __init__(self, rewards=None):
        n_states = 6
        transitions = [[0, 1], [0, 2], [1, 3], [2, 3], [2, 4], [3, 5], [4, 5]]
        if rewards == None:
            rewards = [0, 0, 0, 0, 0, 1]
        env_attributes = EnvironmentAttributes(n_states, transitions, rewards)

        # internally accessible properties
        self._transition_matrix = env_attributes.transition_matrix
        self._rewards = rewards
        self._start_state = 0
        self._state = self._start_state

        # externally accessible properties
        self.q_size = (n_states, self.action_space.n)

    @property
    def action_space(self) -> gym.spaces.Discrete:
        match self._state:
            case 0 | 2:
                return gym.spaces.Discrete(2)
            case 5:
                return None
            case _:
                return gym.spaces.Discrete(1)

    # reset the environment for a new episode
    def reset(self) -> list[int]:
        self._state = self._start_state
        return [self._state]

    # step in the environment
    def step(self, action):
        assert action in self.action_space, "Invalid action"

        # next state
        next_state = self._transition_matrix[self._state].nonzero()[0][action]
        self._state = next_state

        # reward
        reward = self._rewards[self._state]

        # termination
        done = False
        if self.action_space is None:
            done = True

        info = None

        # return next_state as list
        next_state = [next_state] if not isinstance(next_state, list) else next_state

        return next_state, reward, done, info


class Maze:
    """
    A wrapper class for a maze, containing all the information about the maze.
    Initialized to the 2D maze used by Mattar & Daw 2019 by default, however,
    it can be easily adapted to any other maze by redefining obstacles and size.
    """

    def __init__(self):
        # alternative observation space
        # gym.spaces.Box(low=np.array([0,0]), high=np.array([9,6]), dtype=int)

        # maze width and height
        self._world_width = 9
        self._world_height = 6

        # start state
        self._start_state = [0, 2]
        self._state = self.reset()

        # goal state
        self._goal_states = [[8, 0]]

        # all obstacles
        self._obstacles = [[2, 1], [2, 2], [2, 3], [7, 0], [7, 1], [7, 2], [5, 4]]

        # environment switching properties
        self._new_goal = [[6, 0]]
        self._old_obstacles = [[7, 0]]

        # observation space
        self.q_size = (self._world_width, self._world_height, self.action_space.n)

        # max steps
        self.max_steps = float("inf")

    @property
    def action_space(self):
        return gym.spaces.Discrete(4)

    def reset(self):
        self._state = self._start_state
        return self._state

    def step(self, action):
        x, y = self._state

        match action:
            case 0:  # up
                y = max(y - 1, 0)
            case 1:  # down
                y = min(y + 1, self._world_height - 1)
            case 2:  # left
                x = max(x - 1, 0)
            case 3:  # right
                x = min(x + 1, self._world_width - 1)

        # default next state, reward, done, info
        next_state = [x, y]
        reward = -1
        done = False
        info = None

        # unless agent reached a goal state or hit a wall
        if next_state in self._goal_states:
            reward = 10
            done = True
        elif next_state in self._obstacles:
            next_state = self._state

        # set next state as current state, return next state as list
        self._state = next_state
        next_state = [next_state] if not isinstance(next_state, list) else next_state

        return next_state, reward, done, info

    # Implement pre-defined changes: move one wall and reward location
    def switch(self):
        self._obstacles = [x for x in self._obstacles if x not in self._old_obstacles]
        self.GOAL_STATES = self._new_goal
        return


class Memory2AFC:
    """
    2AFC task with 12 options, grouped into 4 sets of 3 options each.
    On each trial, one of the four sets is selected with probability
    p(set), and two options are drawn from it uniformly at random.
    """

    def __init__(
        self,
        episodes_train=510,
        episodes_pmt=1020,
        n_pmt=20,
        learnPMT=False,
        delta_1=4,
        delta_2=1,
        delta_pmt=4,
    ):

        # states, acitons, and probability of occurance of states
        self.n_states = 12 * 3
        self.states = np.arange(self.n_states)
        # self.pstate = [.4]*6 + [.1]*6 + [0]*24

        # actions and size of the q-table
        self.actions = np.arange(12 * 3)  # normal, PMT+, PMT-
        self.q_size = len(self.actions) + 1  # terminal

        # Defining rewards:
        self.delta_1 = delta_1
        self.delta_2 = delta_2
        self.rewards = [
            10 + self.delta_1,
            10,
            10 - self.delta_1,
            10 + self.delta_2,
            10,
            10 - self.delta_2,
            10 + self.delta_1,
            10,
            10 - self.delta_1,
            10 + self.delta_2,
            10,
            10 - self.delta_2,
        ]
        # The steps below shouldn't be necessary, but I'm including them
        self.delta_pmt = delta_pmt
        self.rewards += list(np.array(self.rewards) + self.delta_pmt) + list(
            np.array(self.rewards) - self.delta_pmt
        )

        # Experimental design: selecting the sequence of states
        self.episode = 0
        self.episodes_train = episodes_train
        self.episodes_pmt = episodes_pmt
        self.episodes = episodes_train + episodes_pmt
        self.learnPMT = learnPMT

        # pre-generating the sequence of states
        self.state_distribution = np.append(
            np.repeat(np.arange(6), 4), np.arange(6, 12), axis=0
        )

        self.states_training = np.repeat(
            self.state_distribution, self.episodes_train / len(self.state_distribution)
        )
        np.random.shuffle(self.states_training)
        np.random.shuffle(self.states_training)

        self.states_PMT = np.repeat(
            self.state_distribution, (self.episodes_pmt) / len(self.state_distribution)
        )
        np.random.shuffle(self.states_PMT)
        np.random.shuffle(self.states_PMT)

        self.states_pregen = np.append(self.states_training, self.states_PMT)
        # self.states_pregen = np.append(states_PMT[:(self.episodes-len(self.states_pregen))], self.states_pregen)

        # current state and next states
        self.state = self.states_pregen[self.episode]
        self.next_states = np.array([None] * len(self.states_pregen))

        # setting PMT trial indices and type (+Delta or -Delta)
        self.pmt_trial = np.zeros(len(self.states_pregen))
        self.n_pmt = n_pmt

        for ii in range(12):
            idx_ii = np.where(self.states_pregen == ii)[
                0
            ]  # get index where state == ii
            idx_ii = idx_ii[idx_ii > self.episodes_train]  # throw away training indices
            np.random.shuffle(idx_ii)  # shuffle what's left
            idx_i1 = idx_ii[: int(n_pmt / 2)]  # indices for +Delta PMT trials
            idx_i2 = idx_ii[int(n_pmt / 2) : n_pmt]  # indices for -Delta PMT trials

            # indicate pmt trial and type
            self.pmt_trial[idx_i1] = 1
            self.pmt_trial[idx_i2] = -1

            # for first half: next_state is option vs. +Delta deterministic option
            self.next_states[idx_i1] = self.states_pregen[idx_i1] + 12

            # for second half: -Delta deterministic option
            self.next_states[idx_i2] = self.states_pregen[idx_i2] + 24

    def reset(self, newEpisode=False):
        self.state = self.states_pregen[self.episode]
        if newEpisode:
            self.episode += 1
        if self.episode == self.episodes:
            self.episode -= 1
            # print('Reached final episode.')
        return [self.state]

    @property
    def action_space(self):
        if self.state < 12:
            if self.state % 3 == 0:
                choiceSet = [self.state + 1, self.state + 2]  # 1 v 2; PMT 0
            elif self.state % 3 == 1:
                choiceSet = [self.state - 1, self.state + 1]  # 0 v 2; PMT 1
            else:
                choiceSet = [self.state - 2, self.state - 1]  # 0 v 1; PMT 2
        elif self.state < 24:
            choiceSet = [self.state - 12, self.state]
        elif self.state < 36:
            choiceSet = [self.state - 24, self.state]
        np.random.shuffle(choiceSet)
        return choiceSet

    # point to location in q-table
    def idx(self, state, action):
        idx = action
        return idx

    # step in the environment
    def step(self, action):
        # info: [updateQ?, allocGrad?]
        if self.state < 12:
            info = [True, True]
        else:
            info = [False, self.learnPMT]

        # next state
        if self.state < 12:
            next_state = self.next_states[self.episode]
            self.state = next_state
        else:
            next_state = None
        # self.next_states[self.episode] = None

        # reward
        reward = self.rewards[action]
        if action < 12:
            reward += np.random.randn()  # stochastic rewards

        # termination
        done = True if next_state is None else False

        next_state = [next_state] if not isinstance(next_state, list) else next_state

        return next_state, reward, done, info


class BottleneckTask:
    # @transition_matrix:   State transition matrix
    # @reward_matrix:       Rewards corresponding to state transitions
    # @n_states:            Number of states
    # @states:              State indices
    def __init__(
        self,
        n_stages=2,
        stochastic_rewards=True,
        stochastic_choice_sets=True,
        version=1,
    ):
        self.version = version
        self.start_state = 0
        self.state = self.start_state
        self.stochastic_rewards = stochastic_rewards
        self.stochastic_choice_sets = stochastic_choice_sets

        # Defining one stage of the task (to be repeated)
        self.module = np.mat(
            "1 1 1 1 0 0 0 0 0 0 0;\
             0 0 0 0 1 0 0 0 0 0 0;\
             0 0 0 0 1 0 0 0 0 0 0;\
             0 0 0 0 0 1 0 0 0 0 0;\
             0 0 0 0 0 1 0 0 0 0 0;\
             0 0 0 0 0 0 1 1 0 0 0;\
             0 0 0 0 0 0 0 0 1 1 0;\
             0 0 0 0 0 0 0 0 0 0 1;\
             0 0 0 0 0 0 0 0 0 0 1;\
             0 0 0 0 0 0 0 0 0 0 1;\
             0 0 0 0 0 0 0 0 0 0 1"
        )

        # Concatenating n_stages task stages
        trMat = np.kron(np.eye(n_stages), self.module)

        # Adding first and last states
        self.transition_matrix = np.zeros((trMat.shape[0] + 1, trMat.shape[1] + 1))
        self.transition_matrix[:-1, 1:] = trMat
        self.transition_matrix[-1, -1] = 1

        # Transitions, rewards, states
        self.n_states = len(self.transition_matrix)
        self.states = np.arange(self.n_states)
        self.transitions = np.transpose(self.transition_matrix.nonzero())

        # Defining rewards for version 1:
        # key states:                   if 3 stages:
        # p_visit = (.5, .5, .8, .2)    + (.8, .2)
        # dq      = (40, 20, 25, 40)    + (15, 30)
        rewards1 = np.array([0, 140, 50, 100, 20, 0, 0, 20, -20, 20, 0])
        rewards2 = np.array([0, 60, 0, 20, -20, 0, 0, 20, -5, 20, -20])
        rewards3 = np.array([0, 140, 40, 100, 70, 0, 0, 20, 5, 20, -10])

        if version == 2:
            # p_visit = (.8, .2, .8, .2)    + (.8, .2)
            # dq      = (40, 20, 25, 40)    + (15, 30)
            rewards3 = np.array([0, 140, 50, 100, 20, 0, 0, 20, 10, 20, -10])

        elif version == 3:
            # p_visit = (.6, .4, .8, .2)    + (.8, .2)
            # dq      = (40, 20, 25, 40)    + (15, 30)
            rewards1 = np.array([0, 140, 20, 100, 50, 0, 0, 20, -20, 20, 0])

        if n_stages == 2:
            self.rewards = np.hstack((rewards1, rewards2, 0))

        elif n_stages == 3:
            self.rewards = np.hstack((rewards1, rewards2, rewards3, 0))

        else:
            raise Exception("Current only supporting 2-3 stages")

        # actions available from each state
        self.actions = [
            list(self.transition_matrix[ii].nonzero()[0])
            for ii in range(len(self.transition_matrix))
        ]

        # the size of q value
        self.q_size = len(self.transitions)

    def reset(self):
        self.state = self.start_state
        return self.start_state

    @property
    def action_space(self):
        assert self.state <= self.n_states

        if self.stochastic_choice_sets:

            if np.mod(self.state, 11) != 0:
                # Bottleneck states are 0, 11, 22
                choiceSet = self.actions[self.state]

            elif self.state == 0:
                choiceList = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]

                if self.version == 1:
                    # p_visit(5,6) = (.5, .5)
                    choiceSet = random.choices(
                        choiceList, weights=(1, 1, 1, 2, 1, 2), k=1
                    )[0]

                else:
                    # version 2: p_visit(5,6) = (.8, .2)
                    # version 3: p_visit(5,6) = (.6, .4)    (bec of diff. rewards)
                    choiceSet = random.choices(
                        choiceList, weights=(2, 2, 2, 1, 2, 1), k=1
                    )[0]

            elif self.state == 11:
                # p_visit(16,17) = (.8, .2)
                choiceList = [
                    [12, 13],
                    [12, 14],
                    [12, 15],
                    [13, 14],
                    [13, 15],
                    [14, 15],
                ]
                choiceSet = random.choices(choiceList, weights=(2, 2, 2, 1, 2, 1), k=1)[
                    0
                ]

            elif self.state == 22:
                # p_visit(27,28) = (.8, .2)
                choiceList = [
                    [23, 24],
                    [23, 25],
                    [23, 26],
                    [24, 25],
                    [24, 26],
                    [25, 26],
                ]
                choiceSet = random.choices(choiceList, weights=(2, 2, 2, 1, 2, 1), k=1)[
                    0
                ]

        else:
            choiceSet = self.actions[self.state]

        return choiceSet

    # point to location in q-table
    def idx(self, state, action):
        ii = np.logical_and(
            self.transitions[:, 0] == state, self.transitions[:, 1] == action
        )
        return np.searchsorted(ii, True) - 1

    # step in the environment
    def step(self, action):
        # next state
        next_state = action
        self.state = next_state

        # reward
        reward = self.rewards[next_state]
        if self.stochastic_rewards:
            reward += np.random.randn()

        # termination and info/logs
        done = False
        if next_state == self.n_states - 1:
            done = True
        info = None

        return next_state, reward, done, info


#########################################################################
#                                                                       #
# Other tasks that we tried running DRA on include the planning task    #
# from Huys et al 2015, a T-maze, a wrapper for arbitrary 2D gridwords  #
# which we used to create the Mattar & Daw 2019 maze, and a couple of   #
# mazes we tried for the human experiments with Antonio Rangel but      #
# eventually gave up on. Though they are not being used in the current  #
# experiments, they live below. In order to be used with the current    #
# DynamicResourceAllocator object, they need to be modified slightly    #
# to have the openAI gym-like structure as in the BottleneckTask above. #
#                                                                       #
#########################################################################

"""
Defines the class environment as per Huys-Dayan-Rosier's planning task
& the agent's long-term tabular memories: (s,a), r(s,a), Q(s,a), pi(a|s).
"""


class HuysTask:
    # @transition_matrix:   State transition matrix
    # @reward_matrix:       Rewards corresponding to state transitions
    # @n_states:            Number of states
    # @states:              State indices
    def __init__(self, depth=3, n_states=6):
        self.depth = depth
        transitions = np.mat(
            "0 1 0 1 0 0;\
                            0 0 1 0 1 0;\
                            0 0 0 1 0 1;\
                            0 1 0 0 1 0;\
                            1 0 0 0 0 1;\
                            1 0 1 0 0 0"
        )
        rewards = np.mat(
            "0   140  0   20  0   0; \
                          0   0   -20  0  -70  0; \
                          0   0    0  -20  0  -70;\
                          0   20   0   0  -20  0; \
                         -70  0    0   0   0  -20;\
                         -20  0    20  0   0   0"
        )

        # Setting up the transitions and rewards matrices for the
        # extended state space: 6 -> 6 x T_left
        self.transition_matrix = np.zeros(
            ((depth + 1) * n_states, (depth + 1) * n_states), dtype=int
        )
        self.reward_matrix = np.zeros(
            ((depth + 1) * n_states, (depth + 1) * n_states), dtype=int
        )

        nrows = transitions.shape[0]
        Nrows = self.transition_matrix.shape[0]

        for i in range(nrows, Nrows, nrows):
            self.transition_matrix[i - nrows : i, i : i + nrows] = transitions

        for i in range(nrows, Nrows, nrows):
            self.reward_matrix[i - nrows : i, i : i + nrows] = rewards

        # Transitions, rewards, states
        self.n_states = len(self.transition_matrix)
        self.states = np.arange(self.n_states)
        self.transitions = np.transpose(self.transition_matrix.nonzero())
        self.rewards = np.array(self.reward_matrix[self.reward_matrix != 0])


"""
Custom-made two-step T-maze with 14 states.
"""


class Tmaze:
    # @transition_matrix:   State transition matrix
    # @reward_matrix:       Rewards corresponding to state transitions
    # @n_states:            Number of states
    # @states:              State indices
    def __init__(self, depth=3, n_states=6, gridworld=False):
        self.depth = depth
        self.gridworld = gridworld
        self.transition_matrix = np.mat(
            "0 1 0 0 0 0 0 0 0 0 0 0 0 0;\
                    1 0 1 1 0 0 0 0 0 0 0 0 0 0;\
                    0 1 0 0 1 0 0 0 0 0 0 0 0 0;\
                    0 1 0 0 0 1 0 0 0 0 0 0 0 0;\
                    0 0 1 0 0 0 1 0 0 0 0 0 0 0;\
                    0 0 0 1 0 0 0 1 0 0 0 0 0 0;\
                    0 0 0 0 1 0 0 0 1 0 0 0 0 0;\
                    0 0 0 0 0 1 0 0 0 1 0 0 0 0;\
                    0 0 0 0 0 0 1 0 0 0 1 0 1 0;\
                    0 0 0 0 0 0 0 1 0 0 0 1 0 1;\
                    0 0 0 0 0 0 0 0 1 0 0 0 0 0;\
                    0 0 0 0 0 0 0 0 0 1 0 0 0 0;\
                    0 0 0 0 0 0 0 0 1 0 0 0 0 0;\
                    0 0 0 0 0 0 0 0 0 1 0 0 0 0"
        )
        self.reward_matrix = -self.transition_matrix
        self.reward_matrix[8, 10] = 10
        self.reward_matrix[8, 12] = -5
        self.reward_matrix[9, 11] = -10
        self.reward_matrix[9, 13] = 5

        # Transitions, rewards, states
        self.n_states = len(self.transition_matrix)
        self.states = np.arange(self.n_states)
        self.transitions = np.transpose(self.transition_matrix.nonzero())
        self.rewards = np.array(self.reward_matrix[self.reward_matrix != 0])
