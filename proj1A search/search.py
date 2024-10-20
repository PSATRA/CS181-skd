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
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem):
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
    stack = util.Stack()
    visited = []

    if problem.isGoalState(problem.getStartState()):
        return []

    stack.push((problem.getStartState(), []))

    while True:
        if stack.isEmpty():
            return []

        xy, path = stack.pop()
        visited.append(xy)

        if problem.isGoalState(xy):
            return path

        children = problem.getSuccessors(xy)
        if children is not None:
            for child_info in children:
                if child_info[0] not in visited:
                    virtual_path = path + [child_info[1]]
                    stack.push((child_info[0], virtual_path))


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    visited = []

    if problem.isGoalState(problem.getStartState()):
        return []

    queue.push((problem.getStartState(), []))

    while True:
        if queue.isEmpty():
            return []

        xy, path = queue.pop()
        visited.append(xy)

        if problem.isGoalState(xy):
            return path

        children = problem.getSuccessors(xy)
        if children is not None:
            # for state in queue.list:
            # print(state)
            added_items = (state[0] for state in queue.list)
            for child_info in children:
                if child_info[0] not in visited and child_info[0] not in added_items:
                    virtual_path = path + [child_info[1]]
                    queue.push((child_info[0], virtual_path))


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    pq = util.PriorityQueue()
    visited = []

    if problem.isGoalState(problem.getStartState()):
        return []

    pq.push((problem.getStartState(), []), 0)

    while True:
        if pq.isEmpty():
            return []

        xy, path = pq.pop()
        visited.append(xy)

        if problem.isGoalState(xy):
            return path

        children = problem.getSuccessors(xy)
        if children is not None:
            for child_info in children:
                # heap state: (priority, self.count, item)
                added_items = [state[2][0] for state in pq.heap]

                # not exists in PQ, purely add
                if child_info[0] not in visited and child_info[0] not in added_items:
                    virtual_path = path + [child_info[1]]
                    priority = problem.getCostOfActions(virtual_path)
                    pq.push((child_info[0], virtual_path), priority)

                # exits in PQ, may update priority
                elif child_info[0] not in visited and child_info[0] in added_items:
                    for state in pq.heap:
                        if state[2][0] == child_info[0]:
                            old_priority = problem.getCostOfActions(state[2][1])
                            virtual_path = path + [child_info[1]]
                            priority = problem.getCostOfActions(virtual_path)
                            if priority < old_priority:
                                pq.update((child_info[0], virtual_path), priority)
                                break


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    pq = util.PriorityQueue()
    visited = []

    if problem.isGoalState(problem.getStartState()):
        return []

    pq.push((problem.getStartState(), []), 0)

    while True:
        if pq.isEmpty():
            return []

        xy, path = pq.pop()
        visited.append(xy)

        if problem.isGoalState(xy):
            return path

        children = problem.getSuccessors(xy)
        if children is not None:
            for child_info in children:
                # heap state: (priority, self.count, item)
                added_items = [state[2][0] for state in pq.heap]
                heuristic_value = heuristic(child_info[0], problem)

                # not exists in PQ, purely add
                if child_info[0] not in visited and child_info[0] not in added_items:
                    virtual_path = path + [child_info[1]]
                    priority = problem.getCostOfActions(virtual_path)
                    priority += heuristic_value
                    pq.push((child_info[0], virtual_path), priority)

                # exits in PQ, may update priority
                elif child_info[0] not in visited and child_info[0] in added_items:
                    for state in pq.heap:
                        if state[2][0] == child_info[0]:
                            old_priority = problem.getCostOfActions(state[2][1])
                            virtual_path = path + [child_info[1]]
                            priority = problem.getCostOfActions(virtual_path)
                            priority += heuristic_value
                            if priority < old_priority:
                                pq.update((child_info[0], virtual_path), priority)
                                break


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
