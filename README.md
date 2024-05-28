<div align='center'>
<h1>Local Google Maps</h1>
<img align="right"  src="https://cdn.dribbble.com/users/2256783/screenshots/10769871/media/7ed3a8055730a512ebaf59428cb12227.gif">
<h4> <span> </span> <a href="https://github.com/Rares Bejusca /Google Maps /blob/master/README.md"> 
</div>

# :notebook_with_decorative_cover: Structure 


## &#x1F34D; Goal of the assignment 
Give travel directions in the map by using Dynamic Programming to save partial solutions that are the same and Divide and Conquer to find the fastest path at several levels of abstraction. 
In the end we use Dijkstra Algorithm to find the fastest path efficiency. 


## :star2: About the Project
- Algorithms needed: Devide & Conquer / Breadth First Search
- The assignment is a route planner for a fictional place (cities connected by highways)

There are 2 parts: 
Part 1: shortest path (distance)
    - 1.0: 2 BFS with small and simple grid (Flood-fill algo)
    - 1.1: building a graph from array, reduce size of state-space
    - 1.2: BFS using a weighted graph (array based)
    - 1.3: BFS using a weighted graph & priority queue 

Part2: fastest path (time)
    - 2.0: general BFS where we take max speed into account
    - 2.1: make an aglo. to find the nearest node given the grid & graph 
    - 2.2: split graph into multiple graphs to reduce the state-space (Optimiziation Divide & Conquer). We can chain several optimal paths to find a coordinate to coordinate path. 
    - 2.3: Splitting he graph is not always optimal. Therefore, we can add a parameter for how many solutions you want to find for each separate graph. 
    - 3.0: Put everything together to find the optiaml path from one coordinate to another. 