
import numpy as np
from grid_maker import Map
from assignment2_1000000_3671526_notebook import Graph

# Test case 1: Creating a graph from a map
map_data = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
map_obj = Map(map_data)
graph = Graph(map_obj)
assert len(graph.adjacency_list) == 4  # There should be 4 nodes in the graph

# Test case 2: Checking if a coordinate is in the graph
assert (0, 0) in graph  # (0, 0) is a node in the graph
assert (1, 1) not in graph  # (1, 1) is not a node in the graph

# Test case 3: Getting random nodes from the graph
random_node = graph.get_random_node()
assert random_node in graph.adjacency_list

# Test case 4: Finding edges for each node in the graph
graph.find_edges()
assert len(graph.adjacency_list[random_node]) == 2  # Each node should have 2 edges

# Test case 5: Showing coordinates and edges on a plot
import matplotlib.pyplot as plt
graph.show_coordinates()
graph.show_edges()
plt.show()
import numpy as np
from grid_maker import Map
