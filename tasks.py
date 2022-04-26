from ctypes import Union
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple
import gym.spaces
import numpy as np
import random


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
        self._obstacles = [[2, 1], [2, 2], [2, 3], [7, 0], [7, 1], [7, 2], [4, 4]]

        # environment switching properties
        self._new_goal = [[6, 0]]
        self._old_obstacles = [[7, 0]]

        # observation space
        self.q_size = (self._world_width, self._world_height, self.action_space.n)

        # max steps
        self.max_steps = 200

    @property
    def action_space(self):
        return gym.spaces.Discrete(4)

    def reset(self):
        self._state = self._start_state
        self._step_counter = 0
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
        self._step_counter += 1

        # unless agent reached a goal state or hit a wall
        if next_state in self._goal_states:
            reward = 10
            done = True
        elif next_state in self._obstacles:
            next_state = self._state
        
        # enforce max episode length
        if self._step_counter == 200:
            done = True

        # set next state as current state, return next state as list
        self._state = next_state
        next_state = [next_state] if not isinstance(next_state, list) else next_state

        return next_state, reward, done, info

    # Implement pre-defined changes: move one wall and reward location
    def switch(self):
        self._obstacles = [x for x in self._obstacles if x not in self._old_obstacles]
        self.GOAL_STATES = self._new_goal
        return


def option_choice_set_2afc(state: int):
    """Returns choice set for Memory_2AFC task."""
    if state < 12:
        if state % 3 == 0:
            choice_set = [state + 1, state + 2]  # 1 v 2; PMT 0
        elif state % 3 == 1:
            choice_set = [state - 1, state + 1]  # 0 v 2; PMT 1
        else:
            choice_set = [state - 2, state - 1]  # 0 v 1; PMT 2
    elif state < 24:
        choice_set = [state - 12, state]
    elif state < 36:
        choice_set = [state - 24, state]
    return choice_set

@dataclass
class Memory2AFC:
    """
    2AFC task with 12 options, grouped into 4 sets of 3 options each.
    On each trial, one of the four sets is selected with probability
    p(set), and two options are drawn from it uniformly at random.
    This is an attempt at re-coding it RL style with 36 states and 
    2 actions (left, right) in each state. Reward is a function of 
    state and action.
    """
    n_states = 36
    n_actions = 2
    _delta_1 = 4
    _delta_2 = 1
    _delta_pmt = 4
    _rel_freq = 4
    n_training_episodes = 2040
    n_testing_episodes = 1020
    n_bonus_trials_per_option = 20

    def __post_init__(self):
        # initialize states and q-size
        self._states = np.arange(self.n_states)
        self.q_size = (self.n_states + 1)
        
        # define rewards for "options"
        option_rewards = [
            10 + self._delta_1, 10, 10 - self._delta_1,
            10 + self._delta_2, 10, 10 - self._delta_2,
            10 + self._delta_1, 10, 10 - self._delta_1,
            10 + self._delta_2, 10, 10 - self._delta_2,
        ]        
        good_bonus_option_rewards  = list(np.array(option_rewards) + self._delta_pmt)
        bad_bonus_option_rewards = list(np.array(option_rewards) - self._delta_pmt)
        self.option_rewards = option_rewards + good_bonus_option_rewards + bad_bonus_option_rewards

        # define state distribution (frequency with which they appear)
        self._state_distribution = np.append(
            np.repeat(np.arange(6), self._rel_freq), np.arange(6, 12), axis=0
        )

        # pregenerate episodes
        self._episode = 0
        self._pregenerate_episodes()

        # known option rewards
        self.q_fixed = np.array([False] * 12 + [True] * 24)
        self.q_initial = np.append(self.option_rewards * self.q_fixed, 0)
        self.q_initial = np.array(self.q_initial, dtype=float)
        self.q_fixed = np.append(self.q_fixed, np.array([True]))  # terminal

    @property
    def action_space(self):
        return gym.spaces.Discrete(2)

    @property
    def option_choice_set(self):
        return option_choice_set_2afc(state=self._state)

    def reward(self, action):
        option_chosen = self.option_choice_set[action]
        reward = self.option_rewards[option_chosen]

        # stochastic rewards for the regular options
        if option_chosen < 12:
            reward += np.random.randn()

        return reward
    
    def _generate_episode_sequence(self, n_episodes):
        episode_sequence = np.repeat(self._state_distribution, 
                n_episodes / len(self._state_distribution))
        np.random.shuffle(episode_sequence)
        return episode_sequence

    def _insert_bonus_episodes(self, test_episodes):
        """function to insert bonus trials in given sequence"""
        # for each of the twelve options
        for option_id in range(12):
            # determine and pick relevant trials
            ids = [i for i in range(len(test_episodes)) if test_episodes[i] == option_id]
            
            # randomly select n_bonus_trials_per_option/2 for good and bad bonus options
            np.random.shuffle(ids)            
            ids_plus_Delta = ids[: int(self.n_bonus_trials_per_option / 2)]
            ids_minus_Delta = ids[
                int(self.n_bonus_trials_per_option / 2) : self.n_bonus_trials_per_option
            ]

            # put them together and sort in reverse order
            # so that insertion does not change the indexing
            ids_both_Delta = ids_plus_Delta + ids_minus_Delta
            ids_both_Delta.sort(reverse=True)
            ids.sort()

            # insert bonus trials
            for idx in ids_both_Delta:
                if idx in ids_plus_Delta:
                    test_episodes.insert(idx + 1, test_episodes[idx] + 12)
                elif idx in ids_minus_Delta:
                    test_episodes.insert(idx + 1, test_episodes[idx] + 24)

        return np.array(test_episodes)

    def _pregenerate_episodes(self) -> None:
        # Episodes
        training_episodes = self._generate_episode_sequence(self.n_training_episodes)
        test_episodes = self._generate_episode_sequence(self.n_testing_episodes)
        test_episodes = self._insert_bonus_episodes(list(test_episodes))
        self._episode_list = np.append(training_episodes, test_episodes)
        return

    def reset(self):
        """
        Because this task/expt pre-generates a sequence of episodes 
        that are to be followed in that order, this reset function 
        updates the state to the next one from the list.
        """
        if self._episode < len(self._episode_list):
            self._state = self._episode_list[self._episode]
            return [self._state]

    def step(self, action):
        if self._episode < len(self._episode_list):
            # next state, reward, termination
            next_state = [-1]
            reward = self.reward(action)
            done = True
            info = None

            # update episode counter
            self._episode += 1

            return next_state, reward, done, info
        
        else:
            print(f"Ignoring step calls beyond what the environment allows.")


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


class HuysTask:
    """Huys, ..., Dayan, Rosier 2011 planning task."""
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
