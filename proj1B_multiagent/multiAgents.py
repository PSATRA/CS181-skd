# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhost = childGameState.getGhostPositions()
        newGhostStates = childGameState.getGhostStates()

        "*** YOUR CODE HERE ***"
        food = newFood.asList()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newFoodDistance = [manhattanDistance(f, newPos) for f in food]
        newGhostDistance = [manhattanDistance(ghost, newPos) for ghost in newGhost]

        score = 0

        for foodDistance in newFoodDistance:
            if foodDistance == 0:
                score += 500
            elif foodDistance <= 2:
                score += 2
            elif foodDistance <= 5:
                score += 1

        for i, ghostDistance in enumerate(newGhostDistance):
            if ghostDistance == 0:
                if newScaredTimes[i] > 0:
                    score += 500
                elif newScaredTimes[i] == 0:
                    score -= 10000
            elif ghostDistance <= 2:
                if newScaredTimes[i] > 0:
                    score += 2
                elif newScaredTimes[i] == 0:
                    score -= 5000
            elif ghostDistance <= 5:
                if newScaredTimes[i] > 0:
                    score += 1
                elif newScaredTimes[i] == 0:
                    score -= 2

        return childGameState.getScore() + score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(gameState, depth, agent):
            state = []

            if self.depth == depth or (not gameState.getLegalActions(agent)):
                return [self.evaluationFunction(gameState), None]

            if agent == gameState.getNumAgents() - 1:
                depth += 1
                newAgent = 0
            else:
                newAgent = agent + 1

            for action in gameState.getLegalActions(agent):
                newState = minimax(gameState.getNextState(agent, action), depth, newAgent)

                # initial state of a node (agent)
                if not state:
                    state.append(newState[0])
                    state.append(action)

                else:
                    oldBestScore = state[0]
                    newBestScore = newState[0]

                    # if pacman
                    if agent == 0:
                        if oldBestScore < newBestScore:
                            state[0] = newBestScore
                            state[1] = action

                    # if ghost
                    if agent > 0:
                        if oldBestScore > newBestScore:
                            state[0] = newBestScore
                            state[1] = action

            return state

        return minimax(gameState, 0, 0)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def minimax(gameState, depth, agent, a, b):
            state = []

            if self.depth == depth or (not gameState.getLegalActions(agent)):
                return [self.evaluationFunction(gameState), None]

            if agent == gameState.getNumAgents() - 1:
                depth += 1
                newAgent = 0
            else:
                newAgent = agent + 1

            for action in gameState.getLegalActions(agent):
                # initial state of a node (agent)
                if not state:
                    newState = minimax(gameState.getNextState(agent, action), depth, newAgent, a, b)
                    state.append(newState[0])
                    state.append(action)
                    if agent == 0:
                        a = max(state[0], a)
                    else:
                        b = min(state[0], b)

                else:
                    if a > b:
                        return state

                    oldBestScore = state[0]
                    newState = minimax(gameState.getNextState(agent, action), depth, newAgent, a, b)
                    newBestScore = newState[0]

                    # if pacman
                    if agent == 0:
                        if oldBestScore < newBestScore:
                            state[0] = newBestScore
                            state[1] = action
                            a = max(state[0], a)

                    # if ghost
                    if agent > 0:
                        if oldBestScore > newBestScore:
                            state[0] = newBestScore
                            state[1] = action
                            b = min(state[0], b)

            return state

        return minimax(gameState, 0, 0, float("-inf"), float("inf"))[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def minimax(gameState, depth, agent):
            state = []

            if self.depth == depth or (not gameState.getLegalActions(agent)):
                return [self.evaluationFunction(gameState), None]

            if agent == gameState.getNumAgents() - 1:
                depth += 1
                newAgent = 0
            else:
                newAgent = agent + 1

            for action in gameState.getLegalActions(agent):
                newState = minimax(gameState.getNextState(agent, action), depth, newAgent)

                # initial state of a node (agent)
                if not state:
                    if agent == 0:
                        state.append(newState[0])
                    else:
                        state.append(newState[0] / len(gameState.getLegalActions(agent)))
                    state.append(action)

                else:
                    oldBestScore = state[0]
                    newBestScore = newState[0]

                    # if pacman
                    if agent == 0:
                        if oldBestScore < newBestScore:
                            state[0] = newBestScore
                            state[1] = action

                    # if ghost
                    if agent > 0:
                        state[0] += newState[0] / len(gameState.getLegalActions(agent))
                        state[1] = action

            return state

        return minimax(gameState, 0, 0)[1]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    childGameState = currentGameState
    newPos = childGameState.getPacmanPosition()
    newFood = childGameState.getFood()
    newGhost = childGameState.getGhostPositions()
    newGhostStates = childGameState.getGhostStates()

    "*** YOUR CODE HERE ***"
    food = newFood.asList()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newFoodDistance = [manhattanDistance(f, newPos) for f in food]
    newGhostDistance = [manhattanDistance(ghost, newPos) for ghost in newGhost]

    score = 0

    for foodDistance in newFoodDistance:
        if foodDistance == 0:
            score += 500
        elif foodDistance <= 2:
            score += 2
        elif foodDistance <= 5:
            score += 1

    for i, ghostDistance in enumerate(newGhostDistance):
        if ghostDistance == 0:
            if newScaredTimes[i] > 0:
                score += 500
            elif newScaredTimes[i] == 0:
                score -= 10000
        elif ghostDistance <= 2:
            if newScaredTimes[i] > 0:
                score += 2
            elif newScaredTimes[i] == 0:
                score -= 5000
        elif ghostDistance <= 5:
            if newScaredTimes[i] > 0:
                score += 2
            elif newScaredTimes[i] == 0:
                score -= 2
        elif ghostDistance <= 10:
            if newScaredTimes[i] > 0:
                score += 1

    return childGameState.getScore() + 2 * score

# Abbreviation
better = betterEvaluationFunction
