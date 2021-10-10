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
import random
import util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        # return self.minimax(gameState=gameState)
        best_action = self.max_value(
            gameState=gameState, depth=0, agent_idx=0)[1]
        return best_action
        # util.raiseNotDefined()

    def is_terminal_state(self, gameState, depth, agent_idx):
        """
        Helper function to determine if we reached a leaf node in the state search tree
        """

        if gameState.isWin():
            return gameState.isWin()
        elif gameState.isLose():
            return gameState.isLose()
        elif gameState.getLegalActions(agent_idx) is 0:
            return gameState.getLegalActions(agent_idx)
        elif depth >= self.depth * gameState.getNumAgents():
            return self.depth

    def max_value(self, gameState, depth, agent_idx):
        """
        Helper function to go through the whole game-state tree, all the way to the leaves, to determine the maximizing
        backed up value of a state. 

        In order to get the value and the corresponding action we need to create an iterable object such as a list
        and specify the key with which we make the comparison for the maximum value which is the float value in the first
        position of the tuple hence the idx[0].
        """

        value = (float('-Inf'), None)
        legal_actions = gameState.getLegalActions(agent_idx)
        for action in legal_actions:
            successor_state = gameState.generateSuccessor(agent_idx, action)
            number_of_agents = gameState.getNumAgents()
            expand = depth + 1
            current_player = (depth + 1) % number_of_agents
            value = max([value, (self.value(gameState=successor_state, depth=expand,
                        agent_idx=current_player), action)], key=lambda idx: idx[0])
        return value

    def min_value(self, gameState, depth, agent_idx):
        """
        Helper function to go through the whole game-state tree, all the way to the leaves, to determine the minimizing
        backed up value of a state.  

        In order to get the value and the corresponding action we need to create an iterable object such as a list
        and specify the key with which we make the comparison for the minimum value which is the float value in the first
        position of the tuple hence the idx[0].
        """

        value = (float('+Inf'), None)
        legal_actions = gameState.getLegalActions(agent_idx)
        for action in legal_actions:
            successor_state = gameState.generateSuccessor(agent_idx, action)
            number_of_agents = gameState.getNumAgents()
            expand = depth + 1
            current_player = (depth + 1) % number_of_agents
            value = min([value, (self.value(gameState=successor_state, depth=expand,
                        agent_idx=current_player), action)], key=lambda idx: idx[0])
        return value

    def value(self, gameState, depth, agent_idx):
        """
        Helper function that acts as a dispatcher to the above functions. It determines which agents's turn it is (MAX agent: Pacman, MIN agents: ghosts)
        and traverses the tree to the leaves and backs up the state's utility value.
        """

        if self.is_terminal_state(gameState=gameState, depth=depth, agent_idx=agent_idx):
            return self.evaluationFunction(gameState)
        elif agent_idx is 0:
            return self.max_value(gameState=gameState, depth=depth, agent_idx=agent_idx)[0]
        else:
            return self.min_value(gameState=gameState, depth=depth, agent_idx=agent_idx)[0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        alpha = float('-Inf')
        beta = float('+Inf')
        depth = 0
        best_action = self.max_value(
            gameState=gameState, depth=depth, agent_idx=0, alpha=alpha, beta=beta)
        return best_action[1]
        # util.raiseNotDefined()

    def is_terminal_state(self, gameState, depth, agent_idx):
        """
        Helper function to determine if we reached a leaf node in the state search tree
        """

        if gameState.isWin():
            return gameState.isWin()
        elif gameState.isLose():
            return gameState.isLose()
        elif gameState.getLegalActions(agent_idx) is 0:
            return gameState.getLegalActions(agent_idx)
        elif depth >= self.depth * gameState.getNumAgents():
            return self.depth

    def max_value(self, gameState, depth, agent_idx, alpha, beta):
        """
        Helper function to go through the whole game-state tree, all the way to the leaves, to determine the maximizing
        backed up value of a state.  

        In order to get the value and the corresponding action we need to create an iterable object such as a list
        and specify the key with which we make the comparison for the maximum value which is the float value in the first
        position of the tuple hence the idx[0]. Additionally by using the alpha factor we can prune whole game-state subtrees.
        """

        value = (float('-Inf'), None)
        legal_actions = gameState.getLegalActions(agent_idx)
        for action in legal_actions:
            successor_state = gameState.generateSuccessor(agent_idx, action)
            number_of_agents = gameState.getNumAgents()
            expand = depth + 1
            current_player = expand % number_of_agents
            value = max([value, (self.value(gameState=successor_state, depth=expand,
                        agent_idx=current_player, alpha=alpha, beta=beta), action)], key=lambda idx: idx[0])
            if value[0] > beta:
                return value
            alpha = max(alpha, value[0])
        return value

    def min_value(self, gameState, depth, agent_idx, alpha, beta):
        """
        Helper function to go through the whole game-state tree, all the way to the leaves, to determine the minimizing
        backed up value of a state. 

        In order to get the value and the corresponding action we need to create an iterable object such as a list
        and specify the key with which we make the comparison for the minimum value which is the float value in the first
        position of the tuple hence the idx[0]. Additionally by using the beta factor we can prune whole game-state subtrees.

        """

        value = (float('+Inf'), None)
        legal_actions = gameState.getLegalActions(agent_idx)
        for action in legal_actions:
            successor_state = gameState.generateSuccessor(agent_idx, action)
            number_of_agents = gameState.getNumAgents()
            expand = depth + 1
            current_player = expand % number_of_agents
            value = min([value, (self.value(gameState=successor_state, depth=expand,
                        agent_idx=current_player, alpha=alpha, beta=beta), action)], key=lambda idx: idx[0])
            if value[0] < alpha:
                return value
            beta = min(beta, value[0])
        return value

    def value(self, gameState, depth, agent_idx, alpha, beta):
        """
        Helper function that acts as a dispatcher to the above functions. It determines which agents's turn it is (MAX agent: Pacman, MIN agents: ghosts)
        and traverses the tree to the leaves and backs up the state's utility value.
        """

        if self.is_terminal_state(gameState=gameState, depth=depth, agent_idx=agent_idx):
            return self.evaluationFunction(gameState)
        elif agent_idx is 0:
            return self.max_value(gameState=gameState, depth=depth, agent_idx=agent_idx, alpha=alpha, beta=beta)[0]
        else:
            return self.min_value(gameState=gameState, depth=depth, agent_idx=agent_idx, alpha=alpha, beta=beta)[0]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
