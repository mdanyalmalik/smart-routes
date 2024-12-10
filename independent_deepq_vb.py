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

# Define the independent DQN for marl
# input is the state of the graph and actions of the other players


class DQN(nn.Module):
    def __init__(self, state_size, num_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size+num_actions, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Define the agent class


class Agent:
    def __init__(self, num_players, state_size, num_actions, id):
        self.id = id
        self.num_players = num_players
        self.state_size = state_size
        self.action_size = num_players
        self.model = DQN(state_size, num_actions)
        self.target_model = DQN(state_size, num_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-4)

    def done(self, state):
        return state[-1][self.id] == 1


class IndependentDeepQ:
    def __init__(self, graph, num_players, state_size, num_actions):
        self.num_players = num_players
        self.state_size = state_size
        self.action_size = num_players
        self.agents = [Agent(num_players, state_size, num_actions, i)
                       for i in range(num_players)]
        self.num_actions = num_actions
        self.graph = graph
        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def get_input(self, state, action):
        inp_state = torch.tensor(state.flatten()).float()
        # make one hot vector for action
        inp_action = torch.zeros(self.num_actions)
        inp_action[action] = 1
        return torch.concat((inp_state, inp_action), 0)

    def get_action_sets(self, state):
        action_sets = []
        for i in range(self.num_players):
            action_set = []
            for j in range(state.shape[0]):
                if state[j][i] == 1:
                    action_set += self.graph.get_outbound_links(j)
            action_sets.append(action_set)
        return action_sets

    def get_action_values(self, state, actionsets):
        action_values = []
        for i in range(self.num_players):
            action_values_i = []
            for j in range(self.num_actions):
                # ignore invalid actions
                if self.graph.edges[j] not in actionsets[i]:
                    action_values_i.append(-np.inf)
                    continue

                inp = self.get_input(state, j)
                with torch.no_grad():
                    out = self.agents[i].model(inp)
                out = out.detach().numpy()[0]

                action_values_i.append(out)

            action_values.append(action_values_i)

        return action_values

    def act(self, state):
        # get actions for each player
        actions = []
        actions_index = []
        actionsets = self.get_action_sets(state)
        actionvalues = self.get_action_values(state, actionsets)
        for i in range(self.num_players):
            if self.agents[i].done(state):
                actions.append(None)
                actions_index.append(None)
                continue
            if np.random.rand() <= self.epsilon:
                actions.append(random.choice(actionsets[i]))
                actions_index.append(actions[-1].edge_id)
            else:
                actions.append(self.graph.edges[np.argmax(actionvalues[i])])
                actions_index.append(np.argmax(actionvalues[i]))

        # compute rewards
        rewards = np.zeros(self.num_players)
        players_per_edge = np.zeros(self.graph.num_edges())
        for i in range(self.num_players):
            if self.agents[i].done(state):
                continue
            players_per_edge[actions_index[i]] += 1
        for i in range(self.num_players):
            if self.agents[i].done(state):
                continue
            rewards[i] = -actions[i].weight(players_per_edge[actions_index[i]])

        # find next state
        next_state = np.zeros((self.graph.num_nodes(), self.num_players))
        for i in range(self.num_players):
            if self.agents[i].done(state):
                next_state[-1][i] = 1
                continue
            for j in range(next_state.shape[0]):
                if actions[i].end_node == j:
                    next_state[j][i] = 1

        return actions_index, actions, rewards, next_state

    def get_next_values(self, next_state, i):
        next_values = []
        actionset = self.get_action_sets(next_state)[i]
        for j in range(self.num_actions):
            # ignore invalid actions
            if self.graph.edges[j] not in actionset:
                next_values.append(-np.inf)
                continue

            inp = self.get_input(next_state, j)
            with torch.no_grad():
                out = self.agents[i].target_model(inp)
            out = out.detach().numpy()[0]
            next_values.append(out)

        return torch.tensor(np.array(next_values))

    def done(self, state):
        return (state[-1] == np.ones(self.num_players)).all()

    def step(self, state, actions_index, actions, rewards, next_state):
        for i in range(self.num_players):
            if self.agents[i].done(state):
                continue

            inp = self.get_input(state, actions_index[i])

            if self.agents[i].done(next_state):
                target = torch.tensor([rewards[i]]).float()
            else:
                next_values = self.get_next_values(next_state, i)
                target = torch.tensor([rewards[i] + self.gamma *
                                      torch.max(next_values)]).float()

            self.agents[i].optimizer.zero_grad()
            loss = F.mse_loss(self.agents[i].model(inp), target)
            loss.backward()
            self.agents[i].optimizer.step()

    def compute_paths(self):
        state = np.zeros((self.graph.num_nodes(), self.num_players))
        paths = []
        for i in range(self.num_players):
            state[0][i] = 1
        done = False
        while not done:
            path = []
            done = self.done(state)
            actionsets = self.get_action_sets(state)
            actionvalues = self.get_action_values(state, actionsets)

            for i in range(self.num_players):
                if self.agents[i].done(state):
                    path.append(None)
                    continue
                path.append(self.graph.edges[np.argmax(actionvalues[i])])

            next_state = np.zeros((self.graph.num_nodes(), self.num_players))
            for i in range(self.num_players):
                if self.agents[i].done(state):
                    next_state[-1][i] = 1
                    continue
                next_state[path[i].end_node][i] = 1

            state = next_state

            paths.append(path)

        player_paths = [[] for i in range(self.num_players)]
        for path in paths:
            for i in range(len(path)):
                player_paths[i].append(path[i])

        return player_paths

    def train(self, episodes):
        avg_costs = []
        eps = []
        max_steps = 20
        for e in range(episodes):
            total_cost = 0
            state = np.zeros((self.graph.num_nodes(), self.num_players))
            for i in range(self.num_players):
                state[0][i] = 1
            done = False
            steps = 0
            while not done and steps < max_steps:
                actions_index, actions, rewards, next_state = self.act(state)
                total_cost += -sum(rewards)
                done = self.done(next_state)
                self.step(state, actions_index, actions, rewards, next_state)
                state = next_state
                steps += 1

            # update epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # update target models
            for i in range(self.num_players):
                self.agents[i].target_model.load_state_dict(
                    self.agents[i].model.state_dict())

            avg_cost = total_cost/self.num_players
            avg_costs.append(avg_cost)
            eps.append(e)

            if e % 100 == 0:
                print(f"Episode: {e}, Average cost: {avg_cost}")
                print(f"Epsilon: {self.epsilon}\n")

        print(self.compute_paths())

        return avg_costs, eps

    def test(self, state, actions):
        pass


# testing

graph = create_braes_network()

# state is array of vectors of size num_players for each node
num_players = 10
state = np.zeros((graph.num_nodes(), num_players))
# put all players at node 0

for i in range(num_players):
    state[0][i] = 1


state_size = state.flatten().shape[0]
num_actions = graph.num_edges()
deepq = IndependentDeepQ(graph, num_players, state_size, num_actions)

avg_costs, eps = deepq.train(1000)
# plot costs
plt.plot(eps, avg_costs)
plt.xlabel("Episodes")
plt.ylabel("Average cost")

# on the same graph, plot a lower granularity version of the same data
# by averaging over 100 episodes
avg_costs = np.array(avg_costs)
avg_costs = avg_costs.reshape(-1, 100)
avg_costs = np.mean(avg_costs, axis=1)
eps = np.array(eps)
eps = eps.reshape(-1, 100)
eps = np.mean(eps, axis=1)
plt.plot(eps, avg_costs)
plt.show()
