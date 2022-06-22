from training import Simulator
from models import Agent, MaxEntRL, DRA
from tasks import Environment, Maze
import matplotlib.pyplot as plt
import numpy as np

# define agent, environment, and simulator
env1 = Maze()
env2 = Maze()
agent1 = DRA(q_size=env1.q_size, learning_q=0.1, sigma_base=20, lmda=1)
agent2 = MaxEntRL(q_size=env2.q_size, learning_q=0.1, alpha=5)
simulator1 = Simulator(agent=agent1, env=env1)
simulator2 = Simulator(agent=agent2, env=env2)

# debugging
n_actions = env2.action_space.n
# agent2.q[:, :, :] = 10

for ep in range(400):
    state = env2.reset()
    done = False

    while not done:
        # print(f"step  {step}")
        # print(f"state {state}")
        action, prob_actions, _ = agent2.act(state, n_actions)
        next_state, reward, done, _ = env2.step(action)
        agent2.update_values(state, action, reward, next_state, prob_actions)
        state = next_state

print(np.array(agent2.q, dtype=int))

# parameters
n_epochs: int = 2
n_episodes: int = 250

# 3 runs
episodes = list(range(n_episodes))
rewards1_all_runs, rewards2_all_runs = [], []

for epoch in range(n_epochs):

    # initialize reward list
    rewards1, rewards2 = [], []

    # printing
    print(f"Starting epoch {epoch} for both agents")

    # run 10 episodes
    for ep in range(n_episodes):
        # print(f"Episode {ep}\nAgent q-values:\n{agent.q}\n")
        rewards1 += [simulator1.run_episode()]
        # simulator1.update_agent_noise()
        rewards2 += [simulator2.run_episode()]

    # print
    print(f"DRA average reward: {np.mean(rewards1[-10:])}")
    print(f"MaxEntRL average reward: {np.mean(rewards2[-10:])}\n")

    # update rewards list
    rewards1_all_runs += rewards1
    rewards2_all_runs += rewards2

print(f"Done training!")

# define model names and get their q*-values
model_names = ["DRA", "MaxEntRL"]
qs = [np.max(agent.q, axis=-1) for agent in [agent1, agent2]]

# print them
for model_name, q in zip(model_names, qs):
    print(f"{model_name} q*-values:\n{np.around(q.transpose(),2)}\n")

# set q values for states with obstacles to nans
for i in env1._obstacles:
    for q in qs:
        q[tuple(i)] = np.nan

# plotting
fig, axs = plt.subplots(1, 2)
for ax, q, model_name in zip(axs, qs, model_names):
    ax.imshow(q.transpose())
    ax.set_title(f"{model_name} q-values")
    ax.grid(False)
plt.savefig("./figures/Maze_qs2.svg")

# # DRA's sigma
# sigma = np.mean(agent1.sigma, axis=-1)
# plt.imshow(sigma.transpose())
# plt.title("DRA sigma values")
# plt.colorbar()
# plt.show()


# define function to compute model entropy
from scipy.stats import entropy


def compute_state_entropy(
    agent: Agent, env: Environment, state: list, n_samples: int = 1000
):
    if isinstance(agent, MaxEntRL):
        _, prob_actions, _ = agent.act(state=state, n_actions=4)
        return entropy(prob_actions)

    elif isinstance(agent, DRA):
        action_counts = np.array([0, 0, 0, 0])
        for _ in range(n_samples):
            action, _, _ = agent.act(state=state, n_actions=4)
            action_counts[action] += 1
        return entropy(action_counts / np.sum(action_counts))


def compute_entropy_matrix(agent: Agent, env: Environment):
    entropy_matrix = np.zeros(env.q_size[:-1])

    for row, row_val in enumerate(entropy_matrix):
        for col, value in enumerate(row_val):
            print(f"Processing row {row} col {col}")
            entropy_matrix[row, col] = compute_state_entropy(
                agent=agent, env=env, state=[row, col]
            )

    return entropy_matrix


# computing entropy of both models
es = [compute_entropy_matrix(agent, env1) for agent in [agent1, agent2]]

# set q values for states with obstacles to nans
for i in env1._obstacles:
    for e in es:
        e[tuple(i)] = np.nan

# plotting entropy
fig, axs = plt.subplots(1, 2)
for ax, e, model_name in zip(axs, es, model_names):
    ax.imshow(e.transpose())
    ax.set_title(model_name)
    ax.grid(False)
plt.savefig("./figures/Maze_entropy2.svg")
