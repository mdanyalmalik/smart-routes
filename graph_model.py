import random
import inspect


class Edge:
    def __init__(self, start_node, end_node, weight):
        self.edge_id = None
        self.start_node = start_node
        self.end_node = end_node
        self.weight = weight

    def __repr__(self):
        # get function string
        weight_str = inspect.getsource(self.weight)
        return f"Edge({self.start_node}, {self.end_node}, {weight_str})"


class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.edge_count = 0

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, edge):
        edge.edge_id = self.edge_count
        self.edges.append(edge)
        self.edge_count += 1

    def get_outbound_links(self, node):
        return [edge for edge in self.edges if edge.start_node == node]

    def get_outbound_idxs(self, node):
        return [i for i in range(len(self.edges)) if self.edges[i].start_node == node]

    def num_nodes(self):
        return len(self.nodes)

    def num_edges(self):
        return self.edge_count

    def __repr__(self):
        return f"Nodes: {self.nodes}\nEdges: {self.edges}"


def create_braes_network():
    graph = Graph()

    for i in range(0, 4):
        graph.add_node(i)

    graph.add_edge(Edge(0, 1, lambda x: x))
    graph.add_edge(Edge(0, 2, lambda x: 2))
    graph.add_edge(Edge(1, 2, lambda x: 0))
    graph.add_edge(Edge(1, 3, lambda x: 2.1))
    graph.add_edge(Edge(2, 3, lambda x: x-1))

    return graph
