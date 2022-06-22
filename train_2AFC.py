from training import Simulator
from models import DRA, MaxEntRL
from indexers import memory_2afc_task_indexer
from tasks import Memory2AFC, Memory2AFCmdp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

sns.set()


dfs = []
dfs_all_runs = []
dfs_pmt_all_runs = []

for run in range(3):
    print(f"Run {run+1}/10")

    # define environments
    env1, env2 = Memory2AFC(), Memory2AFC()

    # defining agents
    # agent1 = DRA(q_size=env1.q_size, sigma_base=5, lmda=0.1)
    # agent2 = MaxEntRL_2AFC(q_size=env2.q_size, alpha=4)
    alphas = [0.1, 1, 2, 3, 4, 10]
    envs = [Memory2AFC() for _ in range(len(alphas))]
    agents = [
        MaxEntRL(q_size=envs[0].q_size, alpha=alpha, _index=memory_2afc_task_indexer)
        for alpha in alphas
    ]

    # setting up DRA for this environment
    # agent1._index = staticmethod(memory_2afc_task_indexer)
    # agent1.q = env1.q_initial.copy()
    # agent1.fixed_ids = env1.q_fixed
    # agent2.q[:-1, :] = 6
    # agent2.q[12:, 1] = env1.q_initial[12:].copy()
    # agent2.fixed_ids = [[i, 1] for i in np.nonzero(env1.q_fixed)[0]]
    simulators = []
    for agent, env in zip(agents, envs):
        # agent._index = staticmethod(memory_2afc_task_indexer)
        agent.q = env.q_initial
        agent.fixed_ids = env.q_fixed
        simulators += [Simulator(agent=agent, env=env)]

    # simulators
    # simulator1 = Simulator(agent=agent1, env=env1)
    # simulator2 = Simulator(agent=agent2, env=env2)
    simulators = [Simulator(agent, env) for agent, env in zip(agents, envs)]

    # parameters
    n_epochs: int = 11
    n_episodes: int = 300

    # initializing episodes
    episodes = list(range(n_episodes))
    rewards1_all_runs, rewards2_all_runs = [], []

    for epoch in range(n_epochs):
        # initialize reward list
        rewards1, rewards2 = [], []

        # printing
        print(f"Starting epoch {epoch} for all agents")

        # run 10 episodes
        for ep in range(n_episodes):
            # rewards1 += [simulator1.run_episode()]
            # rewards2 += [simulator2.run_episode()]
            for simulator in simulators:
                _ = simulator.run_episode()

    # printing q-values to debug
    print(f"Done training!\n")

    # print(f"DRA q-values:\n{np.around(agent1.q,2)}\n")
    # print(f"DRA sigma-values:\n{np.around(agent1.sigma,2)}\n")
    # print(f"MaxEntRL q-values:\n{np.around(agent2.q,2)}\n")
    # print(f"MaxEntRL values:\n{np.around(agent2.v,2)}\n")

    # analyzing pmt choices
    # df1 = pd.DataFrame(simulator1.choices)
    # df2 = pd.DataFrame(simulator2.choices)

    # df1["model"] = "DRA"
    # df2["model"] = "MaxEntRL"
    # df1 = df1[1000:2040]
    # df2 = df2[1000:2040]

    # df = pd.concat([df1,df2])

    dfs = [pd.DataFrame(simulator.choices) for simulator in simulators]

    for alpha, df in zip(alphas, dfs):
        df["alpha"] = alpha
        # df = df[1000:2040]

    df = pd.concat(dfs)

    # analysis for pmt trials
    df_pmt = df[df["state"] >= 12].copy()  # filter out non-pmt trials
    df_pmt["set"] = (df_pmt["state"] % 12) // 3 + 1
    df_pmt["good_bonus"] = df_pmt["state"] < 24
    df_pmt["Choice accuracy"] = (df_pmt["good_bonus"] * df_pmt["action"]) | (
        (1 - df_pmt["good_bonus"]) * (1 - df_pmt["action"])
    )

    # analysis for non-pmt trials
    df0 = df.copy()
    df0["set"] = (df0["state"] % 12) // 3 + 1
    df0["Choice accuracy"] = 1 - df0["action"]

    # # add df to the list
    # dfs += [
    #     df.groupby(by=["model", "set"])["Choice accuracy"]
    #     .aggregate("mean")
    #     .reset_index()
    # ]

    # plotting
    # df = pd.concat(dfs)
    # sns.lineplot(x="set", y="Choice accuracy", hue="alpha", data=df.reset_index())
    # plt.title(f"Run {run}")
    # plt.savefig(f"./figures/memory2afc/choice_accuracy_maxentRL_pmt_run_{run}.svg")
    # plt.close()
    dfs_all_runs += [df0]
    dfs_pmt_all_runs += [df_pmt]

df = pd.concat(dfs_all_runs).reset_index()
df_pmt = pd.concat(dfs_pmt_all_runs).reset_index()

# plot
# choice accuracy on pmt trials only
sns.lineplot(x="set", y="Choice accuracy", hue="alpha", data=df_pmt.reset_index())
plt.title(f"MaxEntRL PMT trials; all runs combined")
plt.savefig(
    f"./figures/memory2afc/choice_accuracy_maxentRL_PMT_all_runs_combined_regAgent.svg"
)
plt.close()

# choice accuracy on all trials
sns.lineplot(x="set", y="Choice accuracy", hue="alpha", data=df.reset_index())
plt.title(f"MaxEntRL PMT trials; all runs combined")
plt.savefig(
    f"./figures/memory2afc/choice_accuracy_maxentRL_nonPMT_all_runs_combined_regAgent.svg"
)
plt.close()
