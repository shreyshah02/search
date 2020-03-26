# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    #print("Start:", problem.getStartState())
    #print("Is the start state a goal?", problem.isGoalState(problem.getStartState()))
    #print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    #print(type(problem.getSuccessors(problem.getStartState())[1]))
    #util.raiseNotDefined()

    # getting the initial state coordinates of the problem
    start = problem.getStartState()
    # Initializing the path as an empty list
    # Path consists of actions to reach a particular node from the start node
    # Path will store the required actions to reach the goal state from the start state as found by the search algorithm
    path = []
    # Creating a stack object to store the states on the fringe
    # Stack is used here for fringe as it implements LIFO which is required for DFS
    fringe = util.Stack()
    # Making a list start_state consisting of the coordinates of the start state and the path (which is empty for start
    # state)
    start_state = [start, path]
    # Adding the start_state to the fringe
    fringe.push(start_state)
    # Creating an empty set, explored, to keep track of nodes already expanded by the algorithm thus avoiding multiple
    # expansion of the same node
    explored = set()
    # Running the loop until either failure or the path to the goal state is returned
    while True:
        # Checking if the fringe is empty. If it is empty then no solution found and thus return empty
        if fringe.isEmpty():
            return []
        # Getting the coordinates of the latest state added to the fringe and the path to get to it from the start state
        # from the fringe
        node, path = fringe.pop()
        # Checking if the current node is the goal state
        # If it is then the search algorithm has found the solution, which is the path to get to the current node from
        # start node
        # Hence return the path
        if problem.isGoalState(node):
            return path
        # Add the current node to the explored set
        explored.add(node)
        # Getting the possible successors of the current node
        # getSuccessors returns tuples consisting of  - the coordinates of the successors, action to get to the
        # successor from node and cost of the action for all possible sucessors of the node
        successors = problem.getSuccessors(node)
        # Looping over each successor of the current node
        for child in successors:
            # creating a list of coordinates of child and the path from the start state, consistent with entries in the
            # fringe
            child_state_and_action = [child[0], path + [child[1]]]
            # Checking whether or not the child is in fringe and already explored
            # if the child has not been explored and is not in the fringe then add the child state i.e. coordinates and
            # path to child to the fringe
            if child_state_and_action not in fringe.list and child[0] not in explored:
                fringe.push(child_state_and_action)

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Getting the initial state coordinates of the problem
    start = problem.getStartState()
    # Initializing the path as an empty list
    # Path consists of actions to reach a particular node from the start node
    # Path will store the required actions to reach the goal state from the start state as found by the search algorithm
    path = []
    # Checking if the start node is the goal state or not
    # if yes the return the path which is an empty list right now
    if problem.isGoalState(start):
        return path
    # Creating a Queue object to store the nodes on the fringe
    # Queue is used because it implements FIFO which is essentila for BFS search algorithm
    fringe = util.Queue()
    # Adding the start state and the path to the fringe
    fringe.push([start, path])
    # Initializing the explored as an empty list
    # Explored keeps track of the nodes already expanded so to not expand the same node twice
    explored = []
    # Running the loop until either failure or the path to the goal state is returned
    while True:
        # Checking if the fringe is empty. If it is empty then no solution found and thus return empty
        if fringe.isEmpty():
            return []
        # Getting the coordinates of a oldest state on the fringe and the path to get to it from the start state from
        # the fringe
        node, path = fringe.pop()
        # Getting the coordinates of all the states currently on the fringe
        states_on_fringe = [l[0] for l in fringe.list]
        # Adding the current node to the explored set
        explored.append(node)
        # Checking if the current node is the goal state
        # If it is then the search algorithm has found the solution, which is the path to get to the current node from
        # start node
        # Hence return the path
        if problem.isGoalState(node):
            return path
        # Getting the possible successors of the current node
        # getSuccessors returns tuples consisting of  - the coordinates of the successors, action to get to the
        # successor from node and cost of the action for all possible successors of the node
        successors = problem.getSuccessors(node)
        # Looping over each successor of the current node
        for child in successors:
            # Getting the path to the child node by adding the action from the current node to reach the child to
            # the path from start state of the current node
            path_to_child = path + [child[1]]
            # Checking whether or not the child is in fringe and already explored
            # if the child has not been explored and is not in the fringe then add the child state i.e. coordinates and
            # path to child to the fringe
            if child[0] not in states_on_fringe and child[0] not in explored:
                # if problem.isGoalState(child[0]):
                #     return path_to_child
                fringe.push([child[0], path_to_child])

    #util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Getting the initial state coordinates of the problem
    start = problem.getStartState()
    # Initializing the path as an empty list
    # Path consists of actions to reach a particular node from the start node
    # Path will store the required actions to reach the goal state from the start state as found by the search algorithm
    path = []
    # The cost to get to the start state to the start state which is 0
    # cost_path for each expanded node will keep track of the cost of the path from the start state to the node
    cost_path = 0
    # Creating a PriorityQueue object to store the nodes on the fringe
    # PriorityQueue is used because it allows us to prioritize the nodes in the fringe based on the cost path
    # and allows fast retrieval of the lowest priority node i.e the node with the least path cost
    fringe = util.PriorityQueue()
    # Adding the coordinates of the start state, path to the fringe based on the cost of the path
    fringe.push([start, path], cost_path)
    # Initializing the explored as an empty list
    # Explored keeps track of the nodes already expanded so to not expand the same node twice
    explored = []
    # Running the loop until either failure or the path to the goal state is returned
    while True:
        # Checking if the fringe is empty. If it is empty then no solution found and thus return empty
        if fringe.isEmpty():
            return []
        # Getting the coordinates of a state and the path to get to it from the start state with the lowest cost
        # from the fringe
        node, path = fringe.pop()
        # Checking if the current node is the goal state
        # If it is then the search algorithm has found the solution, which is the path to get to the current node from
        # start node
        # Hence return the path
        if problem.isGoalState(node):
            return path
        # Adding the current node to the explored set
        explored.append(node)
        # Getting the possible successors of the current node
        # getSuccessors returns tuples consisting of  - the coordinates of the successors, action to get to the
        # successor from node and cost of the action for all possible successors of the node
        successors = problem.getSuccessors(node)
        # Looping over each successor of the current node
        for child in successors:
            # Getting the path to the child node by adding the action from the current node to reach the child to
            # the path from start state of the current node
            path_to_child = path + [child[1]]
            # Getting the total cost of the path from the start state to the child node
            cost_to_child = problem.getCostOfActions(path_to_child)
            # Getting the coordinates of all the nodes currently in the fringe and storing them as a list
            states_on_fringe = [x[2][0] for x in fringe.heap]
            # Checking whether or not the child is in fringe and already explored
            # if the child has not been explored and is not in the fringe then add the child state i.e. coordinates and
            # path to child to the fringe
            if child[0] not in explored and child[0] not in states_on_fringe:
                fringe.push([child[0], path_to_child], cost_to_child)
            # Checking to see if a child is not in explored but is in the fringe
            # If this is true then updating the priority of the child node in the fringe if a path with lower cost to
            # the child node has been found
            elif child[0] not in explored and child[0] in states_on_fringe:
                # Getting the coordinates and the path of all the nodes currently on the fringe
                state_path_fringe = [y[2] for y in fringe.heap]
                # Looping over all the nodes currently in the fringe
                for x in state_path_fringe:
                    # Getting the previous path of the current child node
                    if x[0] == child[0]:
                        old_path = x[1]
                # Calculating the cost of the previous path for the child node
                old_path_cost = problem.getCostOfActions(old_path)
                # If the cost of the previous node is greater than the new cost then adding the new cost and the new
                # path of the child node using the update method to the fringe
                if old_path_cost > cost_to_child:
                    fringe.update([child[0], path_to_child], cost_to_child)

    #util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # Getting the initial state coordinates of the problem
    start = problem.getStartState()
    # Initializing the path as an empty list
    # Path consists of actions to reach a particular node from the start node
    # Path will store the required actions to reach the goal state from the start state as found by the search algorithm
    path = []
    # The cost to get to the start state to the start state which is 0
    # cost_path for each expanded node will keep track of the cost of the path from the start state to the node
    path_cost = 0
    # heuristic_cost - The estimated cost to reach the goal state from the start state based on the problem
    heuristic_cost = heuristic(start, problem)
    # Creating a PriorityQueue object to store the nodes on the fringe
    # PriorityQueue is used because it allows us to prioritize the nodes in the fringe based on the cost path
    # and allows fast retrieval of the lowest priority node i.e the node with the least path cost
    fringe = util.PriorityQueue()
    # Adding the coordinates of the start state, path to the fringe based on the cost of the path + the heuristic cost
    fringe.push([start, path], path_cost+heuristic_cost)
    # Initializing the explored as an empty list
    # Explored keeps track of the nodes already expanded so to not expand the same node twice
    explored = []
    while True:
        # Checking if the fringe is empty. If it is empty then no solution found and thus return empty
        if fringe.isEmpty():
            return []
        # Getting the coordinates of a state and the path to get to it from the start state with the lowest cost
        # from the fringe
        node, path = fringe.pop()
        # Checking if the current node is the goal state
        # If it is then the search algorithm has found the solution, which is the path to get to the current node from
        # start node
        # Hence return the path to the current node
        if problem.isGoalState(node):
            return path
        # Add the current node to the explored set
        explored.append(node)
        # Getting the possible successors of the current node
        # getSuccessors returns tuples consisting of  - the coordinates of the successors, action to get to the
        # successor from node and cost of the action for all possible successors of the node
        successors = problem.getSuccessors(node)
        # Looping over each successor of the current node
        for child in successors:
            # Getting the Heuristic cost for the child node
            heuristic_cost = heuristic(child[0], problem)
            # Getting the path to the child node by adding the action from the current node to reach the child to
            # the path from start state of the current node
            path_to_child = path + [child[1]]
            # Getting the total cost of the path from the start state to the child node and adding to it the heuristic
            # cost to get to the goal from child node
            path_to_child_cost = problem.getCostOfActions(path_to_child) + heuristic_cost
            # Getting coordinates of all the nodes currently on the fringe
            states_on_fringe = [x[2][0] for x in fringe.heap]
            # Checking whether or not the child is in fringe and already explored
            # if the child has not been explored and is not in the fringe then add the child state i.e. coordinates and
            # path to child to the fringe
            if child[0] not in explored and child[0] not in states_on_fringe:
                fringe.push([child[0], path_to_child], path_to_child_cost)
            # Checking to see if a child is not in explored but is in the fringe
            # If this is true then updating the priority of the child node in the fringe if a path with lower
            # cost(path cost + heuristic cost) to the child node has been found
            elif child[0] not in explored and child[0] in states_on_fringe:
                # Getting all the nodes (coordinates and the path) currently on the fringe
                state_path_fringe = [x[2] for x in fringe.heap]
                # Looping over the state_path_fringe
                for y in state_path_fringe:
                    # Getting the previous path of the current child node
                    if y[0] == child[0]:
                        old_path = y[1]
                # Getting the total cost of the old path from the start state
                old_path_cost = problem.getCostOfActions(old_path)
                # If the cost of the previous node is greater than the new cost then adding the new cost and the new
                # path of the child node using the update method to the fringe
                if old_path_cost > path_to_child_cost:
                    fringe.update([child[0], path_to_child], path_to_child_cost)

    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
