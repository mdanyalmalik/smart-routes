from itertools import product
from graph_model import create_braes_network

# Create the network
graph = create_braes_network()

# Parameters
num_players = 10
# Each path is represented as a list of edges (edge indices)

# paths for adapted braess network
player_paths = [
    [0, 3],       # Path 1: Sequence of edges [0, 3]
    [0, 2, 4],    # Path 2: Sequence of edges [0, 2, 4]
    [1, 4],       # Path 3: Sequence of edges [1, 4]
]

# Initialize variables
min_cost = float('inf')
min_path_allocation = None

# Generate all possible combinations of paths for players
combinations = list(product(range(len(player_paths)), repeat=num_players))

# Brute force computation of total costs
for combination in combinations:
    # Initialize players per edge
    total_cost = 0
    max_path_length = max(len(path) for path in player_paths)

    # Simulate step by step
    for step in range(max_path_length):
        # Clear edge usage at each time step
        players_per_edge = [0] * graph.num_edges()

        # Count players on each edge for this time step
        for player, path_idx in enumerate(combination):
            path = player_paths[path_idx]
            if step < len(path):  # Player is still traversing their path
                edge_id = path[step]
                players_per_edge[edge_id] += 1

        # Calculate cost for this time step
        for player in range(num_players):
            path = player_paths[combination[player]]
            if step < len(path):  # Player is still traversing their path
                # get player's edge at this time step
                edge_id = path[step]
                # get the number of players on this edge
                players_on_edge = players_per_edge[edge_id]
                # get the cost of this edge
                cost = graph.edges[edge_id].weight(players_on_edge)
                # add the cost to the total cost
                total_cost += cost

    # Update minimum cost and path allocation if this combination is cheaper
    if total_cost < min_cost:
        min_cost = total_cost
        min_path_allocation = combination

# Output the result
print("Minimum Cost:", min_cost)
print("Optimal Path Allocation (path indices):", min_path_allocation)

# Map the allocation to paths for better readability
optimal_paths = [player_paths[path_idx] for path_idx in min_path_allocation]
print("Paths taken by each player:", optimal_paths)
