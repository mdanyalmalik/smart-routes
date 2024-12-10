import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from graph_model import Graph, Edge, create_braes_network

from matplotlib import pyplot as plt

# input is the state of the graph and actions of the other players


class DQN(nn.Module):
    def __init__(self, state_size, num_actions):
        super(DQN, self).__init__()
        # initialise all weights to 0
        self.fc1 = nn.Linear(state_size+2*num_actions, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Agent:
    def __init__(self, id, group_id):
        self.id = id
        self.group_id = group_id
        self.policy = {}

    def done(self, state):
        return state[-1][self.id] == 1

    def update_policy(self, state, action):
        # convert state to nested tuple
        state = tuple([tuple(x) for x in state])
        self.policy[state] = action

    def get_action(self, state, actionset):
        if self.done(state):
            return None
        state = tuple([tuple(x) for x in state])
        if state in self.policy:
            return self.policy[state]
        # return random allowable action
        return random.choice(actionset).edge_id

# Define the group class


class Group:
    def __init__(self, num_players_group, state_size, num_actions, group_id):
        self.num_players_group = num_players_group
        self.group_id = group_id
        self.state_size = state_size
        self.model = DQN(state_size, num_actions)
        self.target_model = DQN(state_size, num_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.players = [Agent(i, group_id) for i in range(
            group_id*num_players_group, (group_id+1)*num_players_group)]


class MeanFieldDeepQ:
    def __init__(self, graph, num_players, state_size, num_actions, num_groups):
        self.num_players = num_players
        self.state_size = state_size
        self.action_size = num_players
        self.num_groups = num_groups
        self.num_players_group = num_players//num_groups
        self.num_actions = num_actions
        self.groups = [Group(self.num_players_group, state_size, num_actions, i)
                       for i in range(num_groups)]
        self.agents = []

        for i in range(num_players):
            for j in range(num_groups):
                if i//self.num_players_group == j:
                    p_idx = i - j*self.num_players_group
                    self.agents.append(self.groups[j].players[p_idx])

        self.graph = graph
        self.memories = [deque(maxlen=2000) for i in range(num_players)]

        # learning parameters
        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99

    def get_player_state(self, state, player):
        return state[:, player]

    def mean_action(self, actions, i):
        # get one hot vector of actions
        actions_oh = np.zeros(self.num_actions)
        num_players_group = self.num_players_group
        group_id = i//num_players_group
        for j in range(group_id*num_players_group, (group_id+1)*num_players_group):
            if actions[j] is None:
                continue
            actions_oh[actions[j]] += 1

        if sum(actions_oh) == 0:
            return actions_oh
        return actions_oh/sum(actions_oh)

    def remember(self, state, actions, rewards, next_state):
        for i in range(self.num_players):
            if self.agents[i].done(state):
                continue
            self.memories[i].append(
                (state, actions[i], self.mean_action(actions, i), rewards[i], next_state))
            # action is index and mean action is one hot vector

    def get_input(self, state, action, mean_action, player):
        player_state = self.get_player_state(state, player)
        inp_state = torch.tensor(player_state).float()
        # make one hot vector for action
        inp_action = torch.zeros(self.num_actions)
        if action is not None:
            inp_action[action] = 1

        inp_mean_action = torch.tensor(mean_action).float()
        return torch.concat((inp_state, inp_action, inp_mean_action), 0)

    def get_action_sets(self, state):
        action_sets = []
        for i in range(self.num_players):
            action_set = []
            for j in range(state.shape[0]):
                if state[j][i] == 1:
                    action_set += self.graph.get_outbound_links(j)
            action_sets.append(action_set)
        return action_sets

    def act(self, state):
        # get actions for each player
        actions = []
        actions_index = []
        actionsets = self.get_action_sets(state)  # action sets for each player
        for i in range(self.num_players):
            if self.agents[i].done(state):
                actions.append(None)
                actions_index.append(None)
                continue
            if np.random.rand() <= self.epsilon:
                actions.append(random.choice(actionsets[i]))
                actions_index.append(actions[-1].edge_id)
            else:
                actions_index.append(
                    self.agents[i].get_action(state, actionsets[i]))
                actions.append(self.graph.edges[actions_index[i]])

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

    def get_next_actions(self, next_state):
        # Compute next actions based on stored policies
        next_actions = []
        actionsets = self.get_action_sets(next_state)
        for j in range(self.num_players):
            if self.agents[j].done(next_state):
                next_actions.append(None)
            else:
                # Use the stored policy to select the action
                next_action_index = self.agents[j].get_action(
                    next_state, actionsets[j])
                next_actions.append(next_action_index)

        return next_actions

    def get_next_values(self, next_state, i, mean_action):
        next_values = []
        for j in range(self.num_actions):
            inp = self.get_input(next_state, j, mean_action, i)
            with torch.no_grad():
                out = self.groups[i].target_model(inp)
            out = out.detach().numpy()[0]
            next_values.append(out)

        return torch.tensor(np.array(next_values))

    def done(self, state):
        return (state[-1] == np.ones(self.num_players)).all()

    def step(self, e):
        for i in range(self.num_players):
            while len(self.memories[i]) > 0:
                # randomly pop from left or right
                if np.random.rand() > 0.5:
                    state, actions_index, mean_action, reward, next_state = self.memories[i].popleft(
                    )
                else:
                    state, actions_index, mean_action, reward, next_state = self.memories[i].pop(
                    )
                group_id = i//self.num_players_group
                if self.agents[i].done(state):
                    continue

                inp = self.get_input(state, actions_index, mean_action, i)
                # get next actions
                next_actions = self.get_next_actions(next_state)
                next_mean_action = self.mean_action(next_actions, i)

                if self.agents[i].done(next_state):
                    target = torch.tensor([reward]).float()
                else:
                    next_values = self.get_next_values(
                        next_state, group_id, next_mean_action)
                    # ignore invalid actions
                    for j in range(len(next_values)):
                        if self.graph.edges[j] not in self.get_action_sets(next_state)[i]:
                            next_values[j] = -np.inf
                    target = torch.tensor([reward + self.gamma *
                                           torch.max(next_values)]).float()

                self.groups[group_id].optimizer.zero_grad()
                loss = F.mse_loss(self.groups[group_id].model(inp), target)
                loss.backward()
                self.groups[group_id].optimizer.step()

                # update policy using model
                vals = []
                for j in range(self.num_actions):
                    if self.graph.edges[j] not in self.get_action_sets(state)[i]:
                        vals.append(-np.inf)
                        continue
                    inp = self.get_input(state, j, mean_action, i)
                    with torch.no_grad():
                        out = self.groups[group_id].model(inp)
                    vals.append(out.detach().numpy()[0])
                vals = np.array(vals)
                self.agents[i].update_policy(state, np.argmax(vals))

                if e % 100 == 0 and len(self.memories[i]) == 0:
                    # get player node based on global state
                    player_node = np.argmax(state[:, i])
                    print(
                        f"Player {i}, State {player_node}, Q values: {vals}, Action: {actions_index}")

    def compute_paths(self):
        state = np.zeros((self.graph.num_nodes(), self.num_players))
        paths = []
        total_cost = 0
        for i in range(self.num_players):
            state[0][i] = 1
        done = False
        while not done:
            path = []
            done = self.done(state)
            actionsets = self.get_action_sets(state)
            for i in range(self.num_players):
                if self.agents[i].done(state):
                    path.append(None)
                    continue
                path.append(
                    self.graph.edges[self.agents[i].get_action(state, actionsets[i])])

            next_state = np.zeros((self.graph.num_nodes(), self.num_players))

            for i in range(self.num_players):
                if self.agents[i].done(state):
                    next_state[-1][i] = 1
                    continue
                for j in range(next_state.shape[0]):
                    if path[i] is None:
                        next_state[-1][i] = 1
                    if path[i].end_node == j:
                        next_state[j][i] = 1

            actions = path
            actions_index = [
                action.edge_id if action is not None else None for action in path]
            rewards = np.zeros(self.num_players)
            players_per_edge = np.zeros(self.graph.num_edges())
            for i in range(self.num_players):
                if self.agents[i].done(state):
                    continue
                players_per_edge[actions_index[i]] += 1
            for i in range(self.num_players):
                if self.agents[i].done(state):
                    continue
                rewards[i] = - \
                    actions[i].weight(players_per_edge[actions_index[i]])

            total_cost += -sum(rewards)

            state = next_state

            paths.append(path)

        player_paths = [[] for i in range(self.num_players)]
        for path in paths:
            for i in range(len(path)):
                player_paths[i].append(path[i])

        return player_paths, total_cost/self.num_players

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
                self.remember(state, actions_index, rewards, next_state)
                total_cost += -sum(rewards)
                done = self.done(next_state)
                state = next_state
                steps += 1

            self.step(e)

            # update epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # update target models
            if e % 10 == 0:
                for i in range(self.num_groups):
                    self.groups[i].target_model.load_state_dict(
                        self.groups[i].model.state_dict())

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


state_size = graph.num_nodes()
num_actions = graph.num_edges()
deepq = MeanFieldDeepQ(graph, num_players, state_size, num_actions, 2)

avg_costs, eps = deepq.train(600)

avg_costs = np.array(avg_costs)
avg_costs = np.clip(avg_costs, 0, 10)
# plot costs
plt.plot(eps, avg_costs)
plt.xlabel("Episodes")
plt.ylabel("Average cost")

# on the same graph, plot a lower granularity version of the same data
# by averaging over 100 episodes
avg_costs = np.array(avg_costs)
avg_costs = avg_costs.reshape(-1, 10)
avg_costs = np.mean(avg_costs, axis=1)
eps = np.array(eps)
eps = eps.reshape(-1, 10)
eps = np.mean(eps, axis=1)
plt.plot(eps[1:], avg_costs[1:])


plt.show()
