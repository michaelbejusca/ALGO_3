############ CODE BLOCK 0 ################

# DO NOT CHANGE THIS CELL.
# THESE ARE THE ONLY IMPORTS YOU ARE ALLOWED TO USE:

import numpy as np
import copy
from grid_maker import Map
from collections import defaultdict, deque

RNG = np.random.default_rng()

############ CODE BLOCK 1 ################

class FloodFillSolver():
    """
    A class instance should at least contain the following attributes after being called:
        :param queue: A queue that contains all the coordinates that need to be visited.
        :type queue: collections.deque
        :param history: A dictionary containing the coordinates that will be visited and as values the coordinate that lead to this coordinate.
        :type history: dict[tuple[int], tuple[int]]
    """
    def __init__(self):
        self.queue = deque()
        self.history = {}
    
    def __call__(self, road_grid, source, destination):
        """
        This method gives a shortest route through the grid from source to destination.
        You start at the source and the algorithm ends if you reach the destination, both coordinates should be included in the path.
        To find the shortest route a version of a flood fill algorithm is used, see the explanation above.
        A route consists of a list of coordinates.

        Hint: The history is already given as a dictionary with as keys the coordinates in the state-space graph and
        as values the previous coordinate from which this coordinate was visited.

        :param road_grid: The array containing information where a house (zero) or a road (one) is.
        :type road_grid: np.ndarray[(Any, Any), int]
        :param source: The coordinate where the path starts.
        :type source: tuple[int]
        :param destination: The coordinate where the path ends.
        :type destination: tuple[int]
        :return: The shortest route, which consists of a list of coordinates and the length of the route.
        :rtype: list[tuple[int]], float
        """
        self.road_grid = road_grid
        self.source = source
        self.destination = destination
        self.main_loop()
        path, length = self.find_path()
        return path, length
        
        #raise NotImplementedError("Please complete this method")       

    def find_path(self):
        """
        This method finds the shortest paths between the source node and the destination node.
        It also returns the length of the path. 
        
        Note, that going from one coordinate to the next has a length of 1.
        For example: The distance between coordinates (0,0) and (0,1) is 1 and 
                     The distance between coordinates (3,0) and (3,3) is 3. 

        The distance is the Manhattan distance of the path.

        :return: A path that is the optimal route from source to destination and its length.
        :rtype: list[tuple[int]], float
        """
        path = []
        current_node = self.destination
        while current_node != self.source:
            path.append(current_node)
            current_node = self.history[current_node]
        path.append(self.source)  # don't forget to add the source
        path.reverse()  # reverse so that path is from source to destination
        return path, len(path) -1

    
    def main_loop(self):
        """
        This method contains the logic of the flood-fill algorithm for the shortest path problem.

        It does not have any inputs nor outputs. 
        Hint, use object attributes to store results.
        """
        self.queue.append(self.source)
        while self.queue:
            current_node = self.queue.popleft()
            if self.base_case(current_node):
                break
            for new_node in self.next_step(current_node):
                if new_node not in self.history:
                    self.queue.append(new_node)
                    self.history[new_node] = current_node

        #raise NotImplementedError("Please complete this method")

    def base_case(self, node):
        """
        This method checks if the base case is reached.

        :param node: The current node/coordinate
        :type node: tuple[int]
        :return: This returns if the base case is found or not
        :rtype: bool
        """
        return node == self.destination
        #raise NotImplementedError("Please complete this method")
        
    def step(self, node, new_node):
        """
        One flood-fill step.

        :param node: The current node/coordinate
        :type node: tuple[int]
        :param new_node: The next node/coordinate that can be visited from the current node/coordinate
        :type new_node: tuple[int]       
        """
        if new_node not in self.history:
            self.queue.append(new_node)
            self.history[new_node] = node
            #raise NotImplementedError("Please complete this method")
        
    def next_step(self, node):
        """
        This method returns the next possible actions.

        :param node: The current node/coordinate
        :type node: tuple[int]
        :return: A list with possible next coordinates that can be visited from the current coordinate.
        :rtype: list[tuple[int]]  
        """
        x, y = node
        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        valid_neighbors = [n for n in neighbors if self.is_valid(n)]
        return valid_neighbors

    def is_valid(self, node):
        """
        This method checks if a node is valid.

        :param node: The current node/coordinate
        :type node: tuple[int]
        :return: This returns if the node is valid or not
        :rtype: bool
        """
        x, y = node
        if x < 0 or y < 0 or x >= self.road_grid.shape[0] or y >= self.road_grid.shape[1]:
            return False  # node is out of bounds
        if self.road_grid[x, y] == 0:
            return False  # node is a house, not a road
        return True

############ CODE BLOCK 10 ################

class GraphBluePrint():
    """
    You can ignore this class, it is just needed due to technicalities.
    """
    def find_nodes(self): pass
    def find_edges(self): pass
    
class Graph(GraphBluePrint):   
    """
    Attributes:
        :param adjacency_list: The adjacency list with the road distances and speed limit.
        :type adjacency_list: dict[tuple[int]: set[edge]], where an edge is a fictional datatype 
                              which is a tuple containing the datatypes tuple[int], int, float
        :param map: The map of the graph.
        :type map: Map
    """
    def __init__(self, map_, start=(0, 0)):
        """
        This function transforms any (city or lower) map into a graph representation.

        :param map_: The map that needs to be transformed.
        :type map_: Map
        :param start: The start node from which we will find all other nodes.
        :type start: tuple[int]
        """
        self.adjacency_list = {}
        self.map = map_
        self.start = start
        
        self.find_nodes()
        self.find_edges()  # This will be implemented in the next notebook cell
        
    def find_nodes(self):
        """
        This method contains a breadth-frist search algorithm to find all the nodes in the graph.
        So far, we called this method `step`. However, this class is more than just the search algorithm,
        therefore, we gave it a bit more descriptive name.

        Note, that we only want to find the nodes, so history does not need to contain a partial path (previous node).
        In `find_edges` (the next cell), we will add edges for each node.
        """
        queue = [self.start]
        visited = set()

        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                neighbours = self.neighbour_coordinates(node)
                self.adjacency_list_add_node(node, neighbours)
                queue.extend(neighbours)
        #raise NotImplementedError("Please complete this method")
                    
    def adjacency_list_add_node(self, coordinate, actions):
        """
        This is a helper function for the breadth-first search algorithm to add a coordinate to the `adjacency_list` and
        to determine if a coordinate needs to be added to the `adjacency_list`.

        Reminder: A coordinate should only be added to the adjacency list if it is a corner, a crossing, or a dead end.
                  Adding the coordinate to the adjacency_list is equivalent to saying that it is a node in the graph.

        :param coordinate: The coordinate that might need to be added to the adjacency_list.
        :type coordinate: tuple[int]
        :param actions: The actions possible from this coordinate, an action is defined as an action in the coordinate state-space.
        :type actions: list[tuple[int]]
        """
        if len(actions) != 4:  # if it's a corner, a crossing, or a dead end
            self.adjacency_list[coordinate] = set(actions)


        #raise NotImplementedError("Please complete this method")
                           
    def neighbour_coordinates(self, coordinate):
        """
        This method returns the next possible actions and is part of the breadth-first search algorithm.
        Similar to `find_nodes`, we often call this method `next_step`.
        
        :param coordinate: The current coordinate
        :type coordinate: tuple[int]
        :return: A list with possible next coordinates that can be visited from the current coordinate.
        :rtype: list[tuple[int]]  
        """
        x, y = coordinate
        neighbours = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]  # assuming 4 directions: up, down, left, right
        valid_neighbours = [n for n in neighbours if 0 <= n[0] < self.map.grid.shape[0] and 0 <= n[1] < self.map.grid.shape[1]]  # assuming map is a 2D grid
        return valid_neighbours

        #raise NotImplementedError("Please complete this method")
    
    def __repr__(self):
        """
        This returns a representation of a graph.

        :return: A string representing the graph object.
        :rtype: str
        """
        # You can change this to anything you like, such that you can easily print a Graph object. An example is already given.
        return repr(dict(sorted(self.adjacency_list.items()))).replace("},", "},\n")

    def __getitem__(self, key):
        """
        A magic method that makes using keys possible.
        This makes it possible to use self[node] instead of self.adjacency_list[node]

        :return: The nodes that can be reached from the node `key`.
        :rtype: set[tuple[int]]
        """
        return self.adjacency_list[key]

    def __contains__(self, key):
        """
        This magic method makes it possible to check if a coordinate is in the graph.

        :return: This returns if the coordinate is in the graph.
        :rtype: bool
        """
        return key in self.adjacency_list

    def get_random_node(self):
        """
        This returns a random node from the graph.
        
        :return: A random node
        :rtype: tuple[int]
        """
        return tuple(RNG.choice(list(self.adjacency_list)))
        
    def show_coordinates(self, size=5, color='k'):
        """
        If this method is used before another method that does a plot, it will be plotted on top.

        :param size: The size of the dots, default to 5
        :type size: int
        :param color: The Matplotlib color of the dots, defaults to black
        :type color: string
        """
        nodes = self.adjacency_list.keys()
        plt.plot([n[1] for n in nodes], [n[0] for n in nodes], 'o', color=color, markersize=size)         # type: ignore

    def show_edges(self, width=0.05, color='r'):
        """
        If this method is used before another method that does a plot, it will be plotted on top.
        
        :param width: The width of the arrows, default to 0.05
        :type width: float
        :param color: The Matplotlib color of the arrows, defaults to red
        :type color: string
        """
        for node, edge_list in self.adjacency_list.items():
            for next_node,_,_ in edge_list:
                plt.arrow(node[1], node[0], (next_node[1] - node[1])*0.975, (next_node[0] - node[0])*0.975, color=color, length_includes_head=True, width=width, head_width=4*width) # type: ignore

############ CODE BLOCK 15 ################
    def find_edges(self):
        """
        This method does a depth-first/brute-force search for each node to find the edges of each node.
        """
        for node in self.adjacency_list:
            self.adjacency_list[node] = set()  # clear the current edges
            for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # for each direction
                next_node, distance = self.find_next_node_in_adjacency_list(node, direction)
                if next_node is not None:  # if a node was found in this direction
                    self.adjacency_list[node].add((next_node, distance, int(self.map[next_node])))

                    
        #raise NotImplementedError("Please complete this method")

    def find_next_node_in_adjacency_list(self, node, direction):
        """
        This is a helper method for find_edges to find a single edge given a node and a direction.

        :param node: The node from which we try to find its "neighboring node" NOT its neighboring coordinates.
        :type node: tuple[int]
        :param direction: The direction we want to search in this can only be 4 values (0, 1), (1, 0), (0, -1) or (-1, 0).
        :type direction: tuple[int]
        :return: This returns the first node in this direction and the distance.
        :rtype: tuple[int], int 
        """
        
        x, y = node
        dx, dy = direction
        distance = 0
        while 0 <= x < self.map.shape[0] and 0 <= y < self.map.shape[1]:  # while the coordinates are within the grid
            if self.map[x, y] != 0:  # if the cell is not a wall
                if (x+dx, y+dy) not in self.adjacency_list:  # if the next cell is not a node
                    return (x, y), distance
                else:
                    x, y = x + dx, y + dy
                    distance += 1
            else:
                x, y = x + dx, y + dy
                distance += 1
        return None, None  # if no node was found in this direction

        #raise NotImplementedError("Please complete this method")

############ CODE BLOCK 120 ################

class FloodFillSolverGraph(FloodFillSolver):
    """
    A class instance should at least contain the following attributes after being called:
        :param queue: A queue that contains all the nodes that need to be visited.
        :type queue: collections.deque
        :param history: A dictionary containing the coordinates that will be visited and as values the coordinate that lead to this coordinate.
        :type history: dict[tuple[int], tuple[int]]
    """
    def __call__(self, graph, source, destination):      
        """
        This method gives a shortest route through the grid from source to destination.
        You start at the source and the algorithm ends if you reach the destination, both nodes should be included in the path.
        A route consists of a list of nodes (which are coordinates).

        Hint: The history is already given as a dictionary with as keys the node in the state-space graph and
        as values the previous node from which this node was visited.

        :param graph: The graph that represents the map.
        :type graph: Graph
        :param source: The node where the path starts.
        :type source: tuple[int]
        :param destination: The node where the path ends.
        :type destination: tuple[int]
        :return: The shortest route, which consists of a list of nodes and the length of the route.
        :rtype: list[tuple[int]], float
        """       
        self.queue = deque([source])
        self.history = {source: None}
        self.destination = destination
        self.source = source
        while self.queue:
            current_node = self.queue.popleft()
            if current_node == destination:
                break
            for next_node in self.next_step(graph, current_node):
                if next_node not in self.history:
                    self.queue.append(next_node)
                    self.history[next_node] = current_node
        return self.find_path(source, destination)
        #raise NotImplementedError("Please complete this method")       

    def find_path(self, source, destination):
        """
        This method finds the shortest paths between the source node and the destination node.
        It also returns the length of the path. 
    
        Note, that going from one node to the next has a length of 1.

        :return: A path that is the optimal route from source to destination and its length.
        :rtype: list[tuple[int]], float
        """
        # Initialize path and current_node
        path = []
        current_node = destination

        # Traverse from destination to source
        while current_node is not None:
            path.append(current_node)
            # Get the node that leads to the current_node
            current_node = self.history.get(current_node)

        # Reverse the path to start from the source
        path.reverse()

        # Calculate the length of the path
        length = len(path) - 1

        return path, length
        #raise NotImplementedError("Please complete this method")
        
     

    def next_step(self, graph, node):
        """
        This method returns the next possible actions.

        :param node: The current node
        :type node: tuple[int]
        :return: A list with possible next nodes that can be visited from the current node.
        :rtype: list[tuple[int]]  
        """
        return [neighbour for neighbour, _, _ in graph.adjacency_list.get(node, [])]
        #raise NotImplementedError("Please complete this method")

############ CODE BLOCK 130 ################

class BFSSolverShortestPath():
    """
    A class instance should at least contain the following attributes after being called:
        :param priorityqueue: A priority queue that contains all the nodes that need to be visited including the distances it takes to reach these nodes.
        :type priorityqueue: list[tuple[tuple(int), float]]
        :param history: A dictionary containing the nodes that will be visited and 
                        as values the node that lead to this node and
                        the distance it takes to get to this node.
        :type history: dict[tuple[int], tuple[tuple[int], int]]
    """   
    def __call__(self, graph, source, destination, vehicle_speed):     

        #comment
        """
        This method gives the shortest route through the graph from the source to the destination node.
        You start at the source node and the comalgorithm ends if you reach the destination node, 
        both nodes should be included in the path.
        A route consists of a list of nodes (which are coordinates).

        :param graph: The graph that represents the map.
        :type graph: Graph
        :param source: The node where the path starts
        :type source: tuple[int] 
        :param destination: The node where the path ends
        :type destination: tuple[int]
        :param vehicle_speed: The maximum speed of the vehicle.
        :type vehicle_speed: float
        :return: The shortest route and the time it takes. The route consists of a list of nodes.
        :rtype: list[tuple[int]], float
        """       
        self.priorityqueue = [(source, 0)]
        self.history = {source: (None, 0)}
        self.destination = destination
        self.graph = graph
        self.source = source
        self.destination = destination
        self.vehicle_speed = vehicle_speed
        
        route, distance = self.find_path()
        
        if distance is not None:
            time = distance / self.vehicle_speed
        else:
            time = None
        return route, time
        
               #:rtype: list[tuple[int]], float

    def add_to_priority_queue(self, distance, node):
        '''
        Insert node at the correct position in the priority queue
        '''
        self.priorityqueue.append((distance, node))
        self.priorityqueue.sort()
    
    def pop_from_priority_queue(self):
        '''
         Remove and return the node with the smallest distance
        '''
        return self.priorityqueue.pop(0)

    def find_path(self):
        """
        This method finds the shortest paths between the source node and the destination node.
        It also returns the length of the path. 
        
        Note, that going from one node to the next has a length of 1.

        :return: A path that is the optimal route from source to destination and its length.
        :rtype: list[tuple[int]], float
        """
        path = []
        current_node = self.destination
        while current_node is not None:
            path.append(current_node)
            if current_node in self.history:
                current_node = self.history[current_node][0]
            else:
                break
        path.reverse()

        return path, self.history.get(self.destination, (None, None))[1]
        #raise NotImplementedError("Please complete this method")       

    def main_loop(self):
        """
        This method contains the logic of the flood-fill algorithm for the shortest path problem.

        It does not have any inputs nor outputs. 
        Hint, use object attributes to store results.
        """
        while self.priorityqueue:
            current_distance, current_node = self.pop_from_priority_queue()
            if self.base_case(current_node):
                break
            elif current_node in self.graph:
                for neighbor in self.graph[current_node]:
                    new_distance = current_distance + 1
                    if neighbor not in self.history or new_distance < self.history[neighbor][1]:
                        self.history[neighbor] = (current_node, new_distance)
                        self.add_to_priority_queue(new_distance, neighbor)
        #raise NotImplementedError("Please complete this method")

    def base_case(self, node):
        """
        This method checks if the base case is reached.

        :param node: The current node
        :type node: tuple[int]
        :return: Returns True if the base case is reached.
        :rtype: bool
        """
        return node == self.destination
        #raise NotImplementedError("Please complete this method")

    def new_cost(self, previous_node, distance, speed_limit):
        """
        This is a helper method that calculates the new cost to go from the previous node to
        a new node with a distance and speed_limit between the previous node and new node.

        For now, speed_limit can be ignored.

        :param previous_node: The previous node that is the fastest way to get to the new node.
        :type previous_node: tuple[int]
        :param distance: The distance between the node and new_node
        :type distance: int
        :param speed_limit: The speed limit on the road from node to new_node. 
        :type speed_limit: float
        :return: The cost to reach the node.
        :rtype: float
        """
        return self.history[previous_node][1] + distance
        #raise NotImplementedError("Please complete this method")

    def step(self, node, new_node, distance, speed_limit):
        """
        One step in the BFS algorithm. For now, speed_limit can be ignored.

        :param node: The current node
        :type node: tuple[int]
        :param new_node: The next node that can be visited from the current node
        :type new_node: tuple[int]
        :param distance: The distance between the node and new_node
        :type distance: int
        :param speed_limit: The speed limit on the road from node to new_node. 
        :type speed_limit: float
        """
        new_cost = self.new_cost(node, distance, speed_limit)
        if new_node not in self.history or new_cost < self.history[new_node][1]:
            self.history[new_node] = (node, new_cost)
            self.add_to_priority_queue(new_cost, new_node)
        #raise NotImplementedError("Please complete this method")
    
    def next_step(self, node):
        """
        This method returns the next possible actions.

        :param node: The current node
        :type node: tuple[int]
        :return: A list with possible next nodes that can be visited from the current node.
        :rtype: list[tuple[int]]  
        """
        return self.graph.neighbors(node)
        #raise NotImplementedError("Please complete this method")

############ CODE BLOCK 200 ################

class BFSSolverFastestPath(BFSSolverShortestPath):
    """
    A class instance should at least contain the following attributes after being called:
        :param priorityqueue: A priority queue that contains all the nodes that need to be visited 
                              including the time it takes to reach these nodes.
        :type priorityqueue: list[tuple[tuple[int], float]]
        :param history: A dictionary containing the nodes that will be visited and 
                        as values the node that lead to this node and
                        the time it takes to get to this node.
        :type history: dict[tuple[int], tuple[tuple[int], float]]
    """   
    def __call__(self, graph, source, destination, vehicle_speed):      
        """
        This method gives a fastest route through the grid from source to destination.

        This is the same as the `__call__` method from `BFSSolverShortestPath` except that 
        we need to store the vehicle speed. 
        
        Here, you can see how we can overwrite the `__call__` method but 
        still use the `__call__` method of BFSSolverShortestPath using `super`.
        """
        self.vehicle_speed = vehicle_speed
        return super(BFSSolverFastestPath, self).__call__(graph, source, destination)

    def new_cost(self, previous_node, distance, speed_limit):
        """
        This is a helper method that calculates the new cost to go from the previous node to
        a new node with a distance and speed_limit between the previous node and new node.

        Use the `speed_limit` and `vehicle_speed` to determine the time/cost it takes to go to
        the new node from the previous_node and add the time it took to reach the previous_node to it..

        :param previous_node: The previous node that is the fastest way to get to the new node.
        :type previous_node: tuple[int]
        :param distance: The distance between the node and new_node
        :type distance: int
        :param speed_limit: The speed limit on the road from node to new_node. 
        :type speed_limit: float
        :return: The cost to reach the node.
        :rtype: float
        """
        raise NotImplementedError("Please complete this method")


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
