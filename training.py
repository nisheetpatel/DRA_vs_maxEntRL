from dataclasses import dataclass
from models import Agent, DRA
from tasks import Environment, Memory2AFC
import numpy as np


def kl_divergence_MVN(m0, S0, m1, S1) -> float:
    """
    KL-divergence from Gaussian m0,S0 to Gaussian m1,S1,
    expressed in nats. Diagonal covariances are assumed.
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # three terms of the KL divergence
    tr_term = np.trace(iS1 @ S0)
    # det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0))
    det_term = np.trace(np.ma.log(S1)) - np.trace(np.ma.log(S0))
    quad_term = diff.T @ np.linalg.inv(S1) @ diff
    
    return 0.5 * (tr_term + det_term + quad_term - N)


@dataclass
class Simulator:
    agent: Agent
    env: Environment

    def __post_init__(self):
        self.choices =  {"state":[], "action":[], "reward":[]}

    def _update_agent_visits(self) -> None:
        """Update visits to state for agent"""
        pass
    
    def record_choices(self, state, action, reward):
        self.choices["state"] += state
        self.choices["action"] += [action]
        self.choices["reward"] += [reward]
        return

    def run_episode(self, updateQ=True):
        # Initializing some variables
        tot_reward, reward = 0, 0
        done = False
        state = self.env.reset()

        while not done:
            n_actions = self.env.action_space.n

            # Determine next action
            action, prob_actions, _ = self.agent.act(state, n_actions)

            # Get next state and reward
            next_state, reward, done, _ = self.env.step(action)

            if updateQ:
                self.agent.update_values(state, action, reward, next_state, prob_actions)
            
            if isinstance(self.env, Memory2AFC):
                # if state[0] >= 12:
                self.record_choices(state, action, reward)
            # Update state and total reward obtained
            state = next_state
            tot_reward += reward

            # Update visits for each idx during episode
            if True: #newEp & allocGrad:
                self._update_agent_visits()

        # allocate resources
        if isinstance(self.agent, DRA):
            if isinstance(self.env, Memory2AFC):
                if (self.env._episode < 2000):
                    self.update_agent_noise()
            else:
                self.update_agent_noise()
            
        return tot_reward

    def update_agent_noise(self):
        """Generate trajectories on policy."""
        
        grads = []

        for _ in range(self.agent.n_trajectories):
            # Initializing some variables
            grad = np.zeros(self.agent.sigma.shape)
            tot_reward, reward = 0, 0
            done = False
            state = self.env.reset()
            r = []

            while not done:
                n_actions = self.env.action_space.n

                # Determine next action
                action, prob_actions, zeta = self.agent.act(state, n_actions)

                # Get next state and reward
                next_state, reward, done, _ = self.env.step(action)
                r.append(reward)

                idx_sa = self.agent._index(state=state, action=action)
                idx_s = self.agent._index(state=state, n_actions=n_actions)

                # advantage function
                psi = self.agent.q[idx_sa] - np.dot(self.agent.q[idx_s], prob_actions)
                
                # gradients
                grad[idx_s] -= (self.agent.beta * zeta * prob_actions) * psi
                grad[idx_sa] += psi * self.agent.beta * zeta[action]      #### CHECK

                # Update state and total reward obtained
                state = next_state
                tot_reward += reward

            # collect sampled stoch. gradients for all trajectories
            grads += [grad]
        
        # Setting fixed and terminal sigmas to sigma_base to avoid
        # divide by zero error; reset to 0 at the end of the loop
        if isinstance(self.env, Memory2AFC):
            self.agent.sigma[self.agent.fixed_ids] = self.agent.sigma_base

        # Compute average gradient across sampled trajs & cost
        grad_cost = self.agent.sigma / (self.agent.sigma_base**2) - 1 / self.agent.sigma
        grad_mean = np.mean(grads, axis=0)

        # Updating sigmas
        self.agent.sigma += self.agent.learning_sigma * (grad_mean - self.agent.lmda * grad_cost)

        # reset the original state
        if isinstance(self.env, Memory2AFC):
            self.env._episode -= self.agent.n_trajectories
            self.agent.sigma[self.agent.fixed_ids] = 0

        return

    def compute_expected_reward(self, n_episodes):
        rewards = []

        for ep in range(n_episodes):
            rewards.append(self.run_episode(updateQ=False))
            
        return np.mean(rewards)

    def compute_cost(self) -> float:
        """compute the cost of representing memories precisely"""
        # define moments of the distributions
        q = self.agent.q.flatten()
        S1 = np.diag(np.square(self.agent.sigma.flatten()))
        S0 = np.diag(np.square(np.ones(len(self.agent.sigma.flatten())) * self.agent.sigma_base))

        return self.agent.lmda * kl_divergence_MVN(q, S1, q, S0)
