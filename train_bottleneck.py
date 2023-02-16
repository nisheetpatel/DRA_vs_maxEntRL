from training import Simulator
from models import DRA, MaxEntRL_bottleneck, bottleneck_task_indexer
from tasks import Bottleneck
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# define environments
# env1, env2 = Bottleneck(), Bottleneck()

# defining agents
# agent1 = DRA(q_size=env1.q_size, sigma_base=35, lmda=1, _index=bottleneck_task_indexer)

n_runs = 3
envs = []
agents = []

# model params
alphas = [5, 50, 100, 250]
sigma_bases = [10, 20, 30, 40]
lmdas = [0.5, 1, 2, 5, 10]

for _ in range(n_runs):
    for alpha in alphas:
        envs += [Bottleneck()]
        agents += [
            MaxEntRL_bottleneck(
                q_size=envs[0].q_size, alpha=alpha, _index=bottleneck_task_indexer
            )
        ]

    for s, l in itertools.product(sigma_bases, lmdas):
        envs += [Bottleneck()]
        agents += [
            DRA(
                q_size=envs[0].q_size,
                sigma_base=s,
                lmda=l,
                _index=bottleneck_task_indexer,
            )
        ]

# setting up DRA for this environment
# agent1._index = staticmethod(bottleneck_task_indexer)

# simulators
# simulator1 = Simulator(agent=agent1, env=env1)
# simulator2 = Simulator(agent=agent2, env=env2)
simulators = [Simulator(agent, env) for agent, env in zip(agents, envs)]

# parameters
n_epochs: int = 10
n_episodes: int = 100

# initializing episodes
episodes = list(range(n_episodes))
# rewards1_all_runs, rewards2_all_runs = [], []


df_all_runs = []

for i, simulator in enumerate(simulators):

    print(f"Starting training for model {i}/{len(simulators)}...")
    print(f"Model: {simulator.agent.__class__.__name__}")

    for epoch in range(n_epochs):
        # initialize reward list
        rewards1, rewards2 = [], []

        # printing
        print(f"Starting epoch {epoch}")

        # run n_episodes
        for ep in range(n_episodes):
        
            _ = simulator.run_episode()
        # rewards1 += [simulator1.run_episode()]
        # rewards2 += [simulator2.run_episode()]

# printing q-values to debug
print("Done training!\n")

# # appending rewards
# rewards1_all_runs += [rewards1]
# rewards2_all_runs += [rewards2]

# collecting data
for simulator in simulators:
    df = pd.DataFrame(simulator.choices)

    df = df[
        (df["state"] == 11)
        | (df["state"] == 12)
        | (df["state"] == 28)
        | (df["state"] == 29)
    ].copy()

    df["old_state"] = df["state"]
    df["state"] = df["state"].map(
        {
            11: "Blue (5)",
            12: "Green (6)",
            28: "Red (16)",
            29: "Yellow (17)",
        }
    )
    df["Choice accuracy"] = 1 - df["action"]

    model_name = simulator.agent.__class__.__name__.split("_")[0]
    df["model"] = model_name

    if model_name == "DRA":
        df["lmda"] = simulator.agent.lmda
        df["sigma_base"] = simulator.agent.sigma_base
    elif model_name == "MaxEntRL":
        df["alpha"] = simulator.agent.alpha

    df_all_runs += [df]


df = pd.concat(df_all_runs).reset_index()
df.to_csv("./figures/data_bottleneck.csv", index=False)

# plotting

# max entropy plot
df_plot = df[df["model"] == "MaxEntRL"].copy().sort_values(by="old_state")

sns.lineplot(x="state", y="Choice accuracy", hue="alpha", data=df_plot)
plt.title("Max-entropy RL")
plt.savefig("./figures/bottleneck/maxEntRL.svg")
plt.close()

# DRA plot
df_plot = df[df["model"] == "DRA"].copy().sort_values(by="old_state")
sns.relplot(
    x="state",
    y="Choice accuracy",
    hue="lmda",
    kind="line",
    col="sigma_base",
    data=df.sort_values(by="old_state"),
)
plt.suptitle("DRA", y=1.025)
plt.savefig("./figures/bottleneck/DRA.svg")
plt.close()

# print(f"DRA q-values:\n{np.around(agent1.q)}\n")
# print(f"DRA sigma-values:\n{np.around(agent1.sigma)}\n")
# print(f"MaxEntRL q-values:\n{np.around(agent2.q)}\n")
# print(f"MaxEntRL values:\n{np.around(agent2.v)}\n")

# # debugging
# step_counter = 0
# done = False
# state = env2.reset()

# while not done:
#     step_counter += 1

#     # stepping
#     n_actions = env2.action_space.n
#     action, prob_actions, _ = agent2.act(state, n_actions)
#     next_state, reward, done, _ = env2.step(action)

#     # printing
#     print(f"Step {step_counter}:")
#     print(f"State = {state}, next state = {next_state}\n")

#     agent2.update_values(state, action, reward, next_state, prob_actions)
#     state = next_state
