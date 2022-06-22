from training import Simulator
from models import DRA, MaxEntRL_bottleneck, bottleneck_task_indexer
from tasks import Bottleneck, option_choice_set_bottleneck
import numpy as np


# define environments
env1, env2 = Bottleneck(), Bottleneck()

# defining agents
agent1 = DRA(q_size=env1.q_size, sigma_base=20, lmda=1)
agent2 = MaxEntRL_bottleneck(q_size=env2.q_size, alpha=4)

# setting up DRA for this environment
agent1._index = staticmethod(bottleneck_task_indexer)

# simulators
simulator1 = Simulator(agent=agent1, env=env1)
simulator2 = Simulator(agent=agent2, env=env2)

# parameters
n_epochs: int = 10
n_episodes: int = 100

# initializing episodes
episodes = list(range(n_episodes))
rewards1_all_runs, rewards2_all_runs = [], []


for epoch in range(n_epochs):
    # initialize reward list
    rewards1, rewards2 = [], []

    # printing
    print(f"Starting epoch {epoch} for both agents")

    # run 10 episodes
    for ep in range(n_episodes):
        rewards1 += [simulator1.run_episode()]
        rewards2 += [simulator2.run_episode()]

# printing q-values to debug
print(f"Done training!\n")

print(f"DRA q-values:\n{np.around(agent1.q)}\n")
print(f"DRA sigma-values:\n{np.around(agent1.sigma)}\n")
print(f"MaxEntRL q-values:\n{np.around(agent2.q)}\n")
print(f"MaxEntRL values:\n{np.around(agent2.v)}\n")


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