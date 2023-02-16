from dataclasses import dataclass
from typing import Protocol, Tuple
import gym.spaces
import numpy as np


class Environment(Protocol):
    """A protocol for an environment that can be used by an RL agent."""
    @property
    def action_space() -> gym.spaces.Discrete:
        ...

    def reset() -> list[int]:
        ...

    def step() -> Tuple[list, int, bool, dict]:
        ...


def setup_transition_matrix(n_states: int, transitions: list[int, int]):
    transition_matrix = np.zeros((n_states, n_states), dtype=int)

    for state, next_state in transitions:
        transition_matrix[state, next_state] = 1

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

        # internally accessible properties
        self._transition_matrix = setup_transition_matrix(n_states, transitions)
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
        reward = -10
        done = False
        info = None
        self._step_counter += 1

        # unless agent reached a goal state or hit a wall
        if next_state in self._goal_states:
            reward = 100
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


def option_choice_set_2afc(state: int) -> list:
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
        self.q_size = self.n_states + 1

        # define rewards for "options"
        option_rewards = [
            10 + self._delta_1,
            10,
            10 - self._delta_1,
            10 + self._delta_2,
            10,
            10 - self._delta_2,
            10 + self._delta_1,
            10,
            10 - self._delta_1,
            10 + self._delta_2,
            10,
            10 - self._delta_2,
        ]
        good_bonus_option_rewards = list(np.array(option_rewards) + self._delta_pmt)
        bad_bonus_option_rewards = list(np.array(option_rewards) - self._delta_pmt)
        self.option_rewards = (
            option_rewards + good_bonus_option_rewards + bad_bonus_option_rewards
        )

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
        self.q_initial[:12] = 10
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
        episode_sequence = np.repeat(
            self._state_distribution, n_episodes / len(self._state_distribution)
        )
        np.random.shuffle(episode_sequence)
        return episode_sequence

    def _insert_bonus_episodes(self, test_episodes):
        """function to insert bonus trials in given sequence"""
        # for each of the twelve options
        for option_id in range(12):
            # determine and pick relevant trials
            ids = [
                i for i in range(len(test_episodes)) if test_episodes[i] == option_id
            ]

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


@dataclass
class Memory2AFCmdp:
    """
    2AFC task with 12 options, grouped into 4 sets of 3 options each.
    On each trial, one of the four sets is selected with probability
    p(set), and two options are drawn from it uniformly at random.
    This is an attempt at re-coding it RL style with 36 states and
    2 actions (left, right) in each state. Reward is a function of
    state and action.
    """

    bonus_options: bool = False
    n_states = 37
    n_actions = 2
    _delta_1 = 4
    _delta_2 = 1
    _delta_pmt = 4
    _rel_freq = 4
    _bonus_option_probability = 0.15

    def __post_init__(self):
        # initialize states and q-size
        self._states = np.arange(self.n_states)
        self.q_size = self.n_states

        # define rewards, state distribution,
        self.option_rewards = self._define_option_rewards()
        self._state_distribution = self._define_state_distribution()
        self.info_for_agent = self._define_info_for_agent()

    def _define_option_rewards(self) -> list:
        # define rewards for each of the twelve "options"
        option_rewards = [
            10 + self._delta_1,
            10,
            10 - self._delta_1,
            10 + self._delta_2,
            10,
            10 - self._delta_2,
            10 + self._delta_1,
            10,
            10 - self._delta_1,
            10 + self._delta_2,
            10,
            10 - self._delta_2,
        ]
        good_bonus_option_rewards = list(np.array(option_rewards) + self._delta_pmt)
        bad_bonus_option_rewards = list(np.array(option_rewards) - self._delta_pmt)
        return option_rewards + good_bonus_option_rewards + bad_bonus_option_rewards

    def _define_state_distribution(self):
        # define array with states acc. to their rel. appearance freq.
        state_distribution = np.append(
            np.repeat(np.arange(6), self._rel_freq), np.arange(6, 12), axis=0
        )
        np.random.shuffle(state_distribution)
        return state_distribution

    def _define_info_for_agent(self):
        """
        The agent needs to know some things about this task, which
        is what this function defines. It returns a dictionary with
        the indices for which agents' q-values should be fixed and
        rewards for known (bonus) options as initial q-values.
        """
        # define quantities to be declared to the agent
        fixed_qs = np.array([False] * 12 + [True] * 24)
        initial_qs = np.append(self.option_rewards * fixed_qs, 0)

        # put them in a dictionary
        info_for_agent = {}
        info_for_agent["initial_qs"] = np.array(initial_qs, dtype=float)
        info_for_agent["fixed_qs"] = np.append(fixed_qs, np.array([True]))

        return info_for_agent

    @property
    def action_space(self) -> gym.spaces:
        return gym.spaces.Discrete(2)

    @property
    def option_choice_set(self) -> list:
        return option_choice_set_2afc(state=self._state)

    def reward(self, action):
        option_chosen = self.option_choice_set[action]
        reward = self.option_rewards[option_chosen]

        # stochastic rewards for the regular options
        if option_chosen < 12:
            reward += np.random.randn()

        return reward

    def reset(self):
        self._state = np.random.choice(self._state_distribution)
        return [self._state]

    def _transition(self):
        # defaults
        next_state = [-1]
        done = True

        # define prob. of bonus options
        if self._state < 6:
            p = self._bonus_option_probability
        elif 6 <= self._state < 12:
            p = self._rel_freq * self._bonus_option_probability
        else:
            return next_state, done

        if np.random.binomial(n=1, p=p):
            done = False
            if np.random.binomial(n=1, p=0.5):
                # good bonus option
                next_state = [self._state + 12]
            else:
                # bad bonus option
                next_state = [self._state + 24]

        return next_state, done

    def step(self, action):
        match self.bonus_options:
            case False:
                next_state = [-1]
                done = True
            case True:
                next_state, done = self._transition()

        reward = self.reward(action)
        info = None

        return next_state, reward, done, info


# transitions for bottleneck class and option_choice_set function
bottleneck_transitions = np.array(
    [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [0, 6],  #
        [1, 7],
        [1, 8],
        [2, 7],
        [2, 9],  # top branch
        [3, 7],
        [3, 10],
        [5, 8],
        [5, 10],  # top branch
        [4, 8],
        [4, 9],
        [6, 9],
        [6, 10],  # bottom branch
        [7, 11],
        [8, 11],
        [9, 12],
        [10, 12],
        [11, 13],
        [11, 14],
        [12, 15],
        [12, 16],
        [13, 17],
        [14, 17],
        [15, 17],
        [16, 17],
    ]
)

bottleneck_transitions = np.vstack(
    (
        bottleneck_transitions,
        bottleneck_transitions + np.max(bottleneck_transitions),
        [34, 34],
    )
)


def option_choice_set_bottleneck(
    state: int, all_transitions: np.array = bottleneck_transitions
):
    row_idx = all_transitions[:, 0] == state
    choice_set = all_transitions[row_idx, 1]
    return list(choice_set)


@dataclass
class Bottleneck:
    n_stages: int = 2
    stochastic_rewards: bool = True
    stochastic_choice_sets: bool = True
    _transitions: np.array = bottleneck_transitions
    _first_transition_weights: tuple = (0.2, 0.2, 0.2, 0.1, 0.2, 0.1)

    def __post_init__(self):
        assert self.n_stages == 2, f"Only 2 stages are currently supported"

        # defining n_options and q_size
        self.n_options = np.max(self._transitions) + 1
        self.q_size = self.n_options

        # key states' properties:      if 3 stages:
        # p_visit = (.8, .2, .8, .2)    + (.8, .2)
        # dq      = (40, 20, 25, 40)    + (15, 30)
        self._transition_matrix = setup_transition_matrix(
            self.n_options, self._transitions
        )
        self.rewards = self._define_rewards()

    def _define_rewards(self):
        zeros = np.array([0, 0, 0, 0, 0, 0, 0])
        rewards1 = np.array([140, 50, 100, 20, 0, 0, 20, -20, 20, -5])
        rewards2 = np.array([60, 0, 20, -20, 0, 0, 20, -5, 20, -20])
        return np.hstack((zeros, rewards1, zeros, rewards2, 0))

    def reset(self):
        self._state = 0
        return [self._state]

    @property
    def choice_set(self):
        return option_choice_set_bottleneck(self._state)

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self.choice_set))

    def _autorun_bottleneck_transition(self):
        """Automatically transition when in state 0."""
        assert (self._state == 0) | (
            self._state == (self.n_options / 2)
        ), f"Can only transition without an action in state 0 or 17, not in state {self._state}"
        return np.random.choice(np.arange(1, 7), p=self._first_transition_weights)

    def step(self, action):
        if (self._state == 0) | (self._state == (self.n_options / 2)):
            next_state = self._autorun_bottleneck_transition()
        else:
            next_state = self._transition_matrix[self._state].nonzero()[0][action]

        # determine termination
        done = False
        if next_state == self.n_options - 1:
            done = True
        info = None

        # reward
        reward = self.rewards[next_state]
        if self.stochastic_rewards & (not done):
            if self._state not in [0, self.n_options / 2]:
                reward += np.random.randn()

        # update state and step count
        self._state = next_state

        # return next_state as list
        next_state = [next_state] if not isinstance(next_state, list) else next_state

        return next_state, reward, done, info


@dataclass
class HuysTask:
    """Huys, ..., Dayan, Rosier 2011 planning task."""

    n_max_steps: int = 3
    n_actions: int = 2

    def __post_init__(self):
        self._transition_matrix = np.mat(
            "0 1 0 1 0 0;\
            0 0 1 0 1 0;\
            0 0 0 1 0 1;\
            0 1 0 0 1 0;\
            1 0 0 0 0 1;\
            1 0 1 0 0 0"
        )
        self._reward_matrix = np.mat(
            "0   140  0   20  0   0; \
             0   0   -20  0  -70  0; \
             0   0    0  -20  0  -70;\
             0   20   0   0  -20  0; \
            -70  0    0   0   0  -20;\
            -20  0    20  0   0   0"
        )

        self.n_states = len(self._transition_matrix)

    @property
    def action_space(self):
        return gym.spaces.Discrete(self.n_actions)

    def reset(self):
        self._step_count = 0
        self._state = np.random.choice(self.n_states)
        return [self._state]

    def step(self, action):
        # next state
        next_state = self._transition_matrix[self._state].nonzero()[1][action]
        reward = self._reward_matrix[self._state, next_state]
        done = False
        if self._step_count == self.n_max_steps:
            done = True
        info = None

        # change state and
        self._state = next_state
        self._step_count += 1

        # return next_state as list
        next_state = [next_state] if not isinstance(next_state, list) else next_state

        return next_state, reward, done, info


class Tmaze:
    """Custom-made two-step T-maze with 14 states."""

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
