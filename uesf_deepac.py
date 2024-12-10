import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from graph_model import Graph, Edge, create_braes_network


from matplotlib import pyplot as plt

plt.style.use('dark_background')


class ActionValueNetwork(nn.Module):
    def __init__(self, state_size, num_actions, hidden_size=8):
        super(ActionValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size+num_actions, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# use log-linear learning to compute policy


def compute_policy(q_values, t):
    # compute the policy for the state
    # q_values is a tensor of size num_actions_state
    # t is the temperature
    # return a tensor of size num_actions_state
    if t == 0:
        # one hot vector
        probs = np.zeros(len(q_values))
        probs[np.argmax(q_values)] = 1
        return probs

    q_values = np.array(q_values)
    q_values -= np.max(q_values)  # Prevent overflow
    probs = np.exp(q_values/t) / np.sum(np.exp(q_values/t))
    return probs


class Agent:
    def __init__(self, num_players, state_size, num_actions, id):
        self.id = id
        self.num_players = num_players
        self.state_size = state_size
        self.action_size = num_players
        self.model = ActionValueNetwork(state_size, num_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def done(self, state):
        return state[-1][self.id] == 1


class UESF:
    def __init__(self, graph, num_players, state_size, num_actions):
        self.num_players = num_players
        self.state_size = state_size
        self.action_size = num_players
        self.agents = [Agent(num_players, state_size, num_actions, i)
                       for i in range(num_players)]
        self.memories = [deque(maxlen=100) for i in range(num_players)]
        self.num_actions = num_actions
        self.graph = graph
        self.tau = 5
        self.tau_min = 0.01
        self.tau_decay = 0.995
        self.gamma = 0.95

    def get_input(self, state, action):
        inp_state = torch.tensor(state.flatten()).float()
        # make one hot vector for action
        inp_action = torch.zeros(self.num_actions)
        inp_action[action] = 1
        return torch.concat((inp_state, inp_action), 0)

    def get_action_sets(self, state):
        action_sets = []
        actions_idxs = []
        for i in range(self.num_players):
            action_set = []
            actions_idx = []
            for j in range(state.shape[0]):
                if state[j][i] == 1:
                    action_set += self.graph.get_outbound_links(j)
                    actions_idx += self.graph.get_outbound_idxs(j)
            actions_idxs.append(actions_idx)
            action_sets.append(action_set)
        return action_sets, actions_idxs

    def get_action_values(self, state, actionsets):
        action_values = []
        for i in range(self.num_players):
            if self.agents[i].done(state):
                action_values.append(None)
                continue
            action_values_i = []
            for j in range(self.num_actions):
                # ignore invalid actions
                if self.graph.edges[j] not in actionsets[i]:
                    continue

                inp = self.get_input(state, j)
                with torch.no_grad():
                    out = self.agents[i].model(inp)
                out = out.detach().numpy()[0]

                action_values_i.append(out)

            action_values.append(action_values_i)

        return action_values

    def act(self, state):
        actions = []
        for i in range(self.num_players):
            if self.agents[i].done(state):
                actions.append(None)
                continue
            action_sets, action_idxs = self.get_action_sets(state)
            action_values = self.get_action_values(state, action_sets)

            action = np.random.choice(
                action_idxs[i], p=compute_policy(action_values[i], self.tau))
            actions.append(action)
        return actions

    def get_rewards(self, state, actions):
        rewards = np.zeros(self.num_players)
        players_per_edge = np.zeros(self.graph.num_edges())
        for i in range(self.num_players):
            if self.agents[i].done(state):
                continue
            edge_id = actions[i]
            edge = self.graph.edges[edge_id]
            players_per_edge[edge_id] += 1

        for i in range(self.num_players):
            if self.agents[i].done(state):
                continue
            edge_id = actions[i]
            edge = self.graph.edges[edge_id]
            old = players_per_edge[edge_id] - 1
            new = players_per_edge[edge_id]
            cost = edge.weight(new)
            toll = old*(edge.weight(new) - edge.weight(old))

            rewards[i] = -cost - toll

        return rewards

    def get_actual_costs(self, state, actions):
        costs = np.zeros(self.num_players)
        players_per_edge = np.zeros(self.graph.num_edges())
        for i in range(self.num_players):
            if self.agents[i].done(state):
                continue
            edge_id = actions[i]
            edge = self.graph.edges[edge_id]
            players_per_edge[edge_id] += 1

        for i in range(self.num_players):
            if self.agents[i].done(state):
                continue
            edge_id = actions[i]
            edge = self.graph.edges[edge_id]
            costs[i] = edge.weight(players_per_edge[edge_id])

        return costs

    def step(self, state, actions):
        rewards = self.get_rewards(state, actions)
        costs = self.get_actual_costs(state, actions)
        next_state = np.zeros((self.graph.num_nodes(), self.num_players))
        for i in range(self.num_players):
            if self.agents[i].done(state):
                next_state[-1][i] = 1
                continue
            edge_id = actions[i]
            edge = self.graph.edges[edge_id]
            next_state[edge.end_node][i] = 1

        return next_state, rewards, costs

    def remember(self, state, actions, rewards, next_state):
        for i in range(self.num_players):
            if self.agents[i].done(state):
                continue
            self.memories[i].append(
                (state, actions[i], rewards[i], next_state))

    def display_paths(self, tau_new=0):
        state = np.zeros((self.graph.num_nodes(), self.num_players))
        for j in range(self.num_players):
            state[0][j] = 1
        tau = self.tau
        self.tau = tau_new
        t = 0
        cost = 0
        while not all([self.agents[i].done(state) for i in range(self.num_players)]):
            actions = self.act(state)
            next_state, rewards, costs = self.step(state, actions)
            print(f"Step {t}")

            players_per_edge = np.zeros(self.graph.num_edges())
            for i in range(self.num_players):
                if self.agents[i].done(state):
                    continue
                edge_id = actions[i]
                players_per_edge[edge_id] += 1

            for i in range(self.num_players):
                if self.agents[i].done(state):
                    continue
                edge_id = actions[i]
                edge = self.graph.edges[edge_id]
                print(f"Player {i} takes {edge}")

                old = players_per_edge[edge_id] - 1
                new = players_per_edge[edge_id]
                cost_i = edge.weight(new)
                toll = old*(edge.weight(new) - edge.weight(old))

                print(f"Player {i} takes {edge}, cost: {cost_i}, toll: {toll}")
                print(f"Players per edge: {players_per_edge}")
                print("")

            state = next_state
            t += 1
            cost += sum(costs)

        print(f"Avg cost: {cost/self.num_players}")
        self.tau = tau

    def train(self, episodes):
        eps = []
        avg_costs = []
        max_steps = 20
        for episode in range(episodes):
            state = np.zeros((self.graph.num_nodes(), self.num_players))
            for i in range(self.num_players):
                state[0][i] = 1
            done = False
            step = 0
            cost = 0
            while not done and step < max_steps:
                actions = self.act(state)
                next_state, rewards, costs = self.step(state, actions)
                self.remember(state, actions, rewards, next_state)
                state = next_state
                step += 1
                done = all([self.agents[i].done(state)
                           for i in range(self.num_players)])
                cost += sum(costs)

            for i in range(self.num_players):
                while len(self.memories[i]) > 0:
                    # randomly sample from left or right
                    if random.random() < 0.5:
                        state, action, reward, next_state = self.memories[i].pop(
                        )
                    else:
                        state, action, reward, next_state = self.memories[i].popleft(
                        )
                    if self.agents[i].done(state):
                        continue

                    inp = self.get_input(state, action)
                    target = reward
                    if not self.agents[i].done(next_state):
                        next_values = self.get_action_values(
                            next_state, self.get_action_sets(next_state)[0])
                        target += self.gamma * np.max(next_values[i])

                    self.agents[i].optimizer.zero_grad()
                    # compute loss
                    loss = F.mse_loss(self.agents[i].model(
                        inp), torch.tensor([target]).float())
                    loss.backward()
                    self.agents[i].optimizer.step()

            avg_cost = cost/self.num_players

            eps.append(episode)
            avg_costs.append(avg_cost)

            if self.tau > self.tau_min:
                self.tau *= self.tau_decay

            if episode % 100 == 0:
                print(f"Episode: {episode}, Cost: {avg_cost}, Tau: {self.tau}")

        self.display_paths()

        return eps, avg_costs


graph = create_braes_network()

# state is array of vectors of size num_players for each node
num_players = 10
state = np.zeros((graph.num_nodes(), num_players))
# put all players at node 0

for i in range(num_players):
    state[0][i] = 1


state_size = state.flatten().shape[0]
num_actions = graph.num_edges()
ac = UESF(graph, num_players, state_size, num_actions)

eps, avg_costs = ac.train(1000)

plt.plot(eps, avg_costs)
plt.xlabel("Episodes")
plt.ylabel("Avg Cost")
plt.show()
