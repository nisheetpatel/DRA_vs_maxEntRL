from training import Simulator
from models import Agent, MaxEntRL, DRA
from tasks import Environment, ZiebartTask
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# define function to compute model entropy
from scipy.stats import entropy


def compute_state_entropy(
    agent: Agent, env: Environment, state: list, n_samples: int = 1000
):
    if state[0] == 5:
        return 0
    else:
        env._state = state[0]
        n_actions = env.action_space.n

    if isinstance(agent, MaxEntRL):
        _, prob_actions, _ = agent.act(state=state, n_actions=n_actions)
        return entropy(prob_actions)

    elif isinstance(agent, DRA):
        action_counts = np.zeros(n_actions)
        for _ in range(n_samples):
            action, _, _ = agent.act(state=state, n_actions=n_actions)
            action_counts[action] += 1
        return entropy(action_counts / np.sum(action_counts))


def compute_entropy_matrix(agent: Agent, env: Environment):
    entropy_matrix = np.zeros(env.q_size[:-1])

    for row, row_val in enumerate(entropy_matrix):
        entropy_matrix[row] = compute_state_entropy(agent=agent, env=env, state=[row])

    return entropy_matrix


alphas = [0.1, 0.25, 0.5, 1, 2, 5, 10]
n_runs = 5
results = []

# simulations
for alpha in alphas:

    # printing
    print(f"\nalpha = {alpha}")

    for run in range(n_runs):
        print(f"\trun {run+1}/{n_runs}")

        # define agent, environment, and simulator
        env1 = ZiebartTask()
        env2 = ZiebartTask()
        agent1 = DRA(q_size=env1.q_size, learning_q=0.1, lmda=alpha)
        agent2 = MaxEntRL(q_size=env2.q_size, learning_q=0.1, alpha=alpha)
        simulator1 = Simulator(agent=agent1, env=env1)
        simulator2 = Simulator(agent=agent2, env=env2)

        # parameters
        n_epochs: int = 3
        n_episodes: int = 100

        # 3 runs
        episodes = list(range(n_episodes))
        rewards1_all_runs, rewards2_all_runs = [], []

        betas = np.linspace(0, 10, n_epochs)

        for epoch in range(n_epochs):

            agent1.beta = betas[epoch]

            # run 10 episodes
            for ep in range(n_episodes):
                # print(f"Episode {ep}\nAgent q-values:\n{agent.q}\n")
                simulator1.run_episode()
                simulator2.run_episode()

        # define model names and get their q*-values
        model_names = ["DRA", "MaxEntRL"]
        qs = [agent.q for agent in [agent1, agent2]]

        # computing entropy of both models and printing them
        es = [compute_entropy_matrix(agent, env1) for agent in [agent1, agent2]]
        model_entropies = np.around(es, 2)

        # keeping the ones for states 0 & 2
        delta_qs = [
            float(np.diff(qs[0])[0]),
            float(np.diff(qs[1])[0]),
            float(np.diff(qs[0])[2]),
            float(np.diff(qs[1])[2]),
        ]
        model_entropies = (model_entropies[:, (0, 2)]).transpose().flatten()

        result_run = pd.DataFrame(
            {
                "Model": model_names * 2,
                "State": [0, 0, 2, 2],
                "Stakes": delta_qs,
                "Entropy": model_entropies,
                "Cost": [agent1.lmda, agent2.alpha] * 2,
                "Run": [run] * 4,
            }
        )
        results += [result_run]


# concatenate all results into one dataframe
df = pd.concat(results).reset_index()

# Stakes
sns.catplot(
    x="State",
    y="Stakes",
    hue="Cost",
    col="Model",
    capsize=0.2,
    palette="YlGnBu_d",
    height=6,
    aspect=0.75,
    kind="point",
    data=df,
)
plt.savefig("./figures/ziebart/stakes.svg")


# Entropy
sns.catplot(
    x="State",
    y="Entropy",
    hue="Cost",
    col="Model",
    capsize=0.2,
    palette="OrRd_d",
    height=6,
    aspect=0.75,
    kind="point",
    data=df,
)
plt.savefig("./figures/ziebart/entropy.svg")
