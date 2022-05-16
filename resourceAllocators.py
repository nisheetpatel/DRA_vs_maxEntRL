from dataclasses import dataclass
from tasks import Environment
from typing import Protocol
import numpy as np
from models import DRA, Agent

@dataclass
class MemoryResourceAllocator(Protocol):
    agent: Agent
    env: Environment

    def update(self, sigma: np.array) -> np.array:
        ...

@dataclass
class DynamicRA:
    agent = DRA
    n_traj = 10

    def update(self, sigma: np.array) -> np.array:
        grads = []

        for _ in range(int(self.n_traj)):
            # Initialising some variables
            grad = np.zeros(self.sigma.shape)
            done = False
            tot_reward, reward = 0, 0
            state = self.env.reset()  # newEp=False by default
            r = []

            while not done:
                # Determine next action
                action, z, prob_a, action_space = self.act(state)

                # Get next state and reward
                s1, reward, done, info = self.env.step(action)
                allocGrad = info[1]
                r.append(reward)

                # find pointers to q-table
                idx = self.idx(state, action)
                id_all = [self.idx(state, a) for a in action_space]

                if self.gradient == "A":
                    psi = self.q[idx] - np.mean(self.q[id_all])
                elif self.gradient == "Q":
                    psi = self.q[idx]
                elif self.gradient == "R":
                    psi = 1

                # gradients
                grad[id_all] -= (self.beta * z * prob_a) * psi
                grad[idx] += psi * (
                    self.beta * z[np.array(action_space) == action]
                )

                # Update state for next step, add total reward
                state = s1
                tot_reward += reward

            if self.gradient == "R":
                rturn = np.sum(r)
                grads += [np.dot(rturn, grad)]
            else:
                # Collect sampled stoch. gradients for all trajs
                grads += [grad]
                # reward_list.append(tot_reward)

        # Setting fixed and terminal sigmas to sigmaBase to avoid
        # divide by zero error; reset to 0 at the end of the loop
        self.sigma[self.fixed_ids] = self.sigmaBase
        self.sigma[-1] = self.sigmaBase

        # Compute average gradient across sampled trajs & cost
        grad_cost = self.sigma / (self.sigmaBase**2) - 1 / self.sigma
        grad_mean = np.mean(grads, axis=0)

        # Updating sigmas
        self.sigma += self.learning_sigma * (grad_mean - self.lmda * grad_cost)

        return self.sigma

class EqualPrecisionRA:
    pass

class FrequencyBasedRA:
    pass

class StakesBasedRA:
    pass

class NoRA:
    def update(self, sigma:np.array) -> np.array:
        return sigma