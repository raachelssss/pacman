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

    """


    def getAction(self, gameState: GameState):
        """
        getAction chooses among the best options according to the evaluation function.
        getAction takes a GameState and returns some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        score_list = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(score_list)
        bestIndices = [index for index in range(len(score_list)) if score_list[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        pacfoodDist = [manhattanDistance(newPos, food) for food in newFood.asList()]
        pacghostDist = [manhattanDistance(newPos, ghostPos) for ghostPos in successorGameState.getGhostPositions()]
        food = 0
        if pacfoodDist:
            food = 1/min(pacfoodDist)
        return food + 3*max(pacghostDist) - 3*min(pacghostDist) + 5*successorGameState.getScore()

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

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def minmaxValue(self, gameState, agentIndex, depth):
            if gameState.isWin() or gameState.isLose() or depth == self.depth :
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                actionVal = {action:float('-inf') for action in gameState.getLegalActions(0)}
                for action in actionVal.keys():
                    actionVal[action] = minmaxValue(self, gameState.generateSuccessor(0, action), 1, depth)
                return list(sorted(actionVal.items(), key=lambda x: x[1], reverse=True))[0][1]

            else:
                next = agentIndex + 1
                if next == gameState.getNumAgents() or next == 0:
                    next = 0
                    depth += 1
                actionVal = {action:float('inf') for action in gameState.getLegalActions(agentIndex)}
                for action in actionVal.keys():
                    actionVal[action] = minmaxValue(self, gameState.generateSuccessor(agentIndex, action), next, depth)
                return list(sorted(actionVal.items(), key=lambda x: x[1]))[0][1]

        action_value = {action:float('-inf') for action in gameState.getLegalActions(0)}

        for action in gameState.getLegalActions(0):
            action_value[action] = max(action_value[action], minmaxValue(self, gameState.generateSuccessor(0, action), 1, 0))
        return list(sorted(action_value.items(), key=lambda x: x[1], reverse=True))[0][0]






class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    The minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def alpha_beta(self, gameState, agentIndex, depth, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                actionVal = {action: float('-inf') for action in gameState.getLegalActions(0)}
                v = float('-inf')
                for action in actionVal.keys():
                    # actionVal[action] = max(actionVal[action], alpha_beta(self, gameState.generateSuccessor(0, action), 1, depth, alpha, beta))
                    v = max(v, alpha_beta(self, gameState.generateSuccessor(0, action), 1, depth, alpha, beta))

                    if v >= beta:
                        return v
                    alpha = max(alpha, v)
                if depth > 1:
                    return v
                return v
                # return v

            else:
                # next = agentIndex + 1
                # if next == gameState.getNumAgents():
                #     next = 0
                # if next == 0:
                #     depth += 1
                actionVal = {action: float('inf') for action in gameState.getLegalActions(agentIndex)}
                v = float('inf')
                for action in actionVal.keys():
                    next = agentIndex + 1
                    if next == gameState.getNumAgents():
                        if depth < self.depth:
                            v = alpha_beta(self, gameState.generateSuccessor(agentIndex, action),
                                           agentIndex , depth + 1, alpha, beta)
                        else:
                            v = self.evaluationFunction(gameState.generateSuccessor(agentIndex, action))
                    else:
                        v = min(v, alpha_beta(self, gameState.generateSuccessor(agentIndex, action),
                                          agentIndex + 1 , depth, alpha, beta))

                    if v <= alpha:
                        return v
                    beta = min(v, beta)
                    # if beta <= alpha:
                    #     return v
                return v

        actionVal = {action: float('-inf') for action in gameState.getLegalActions(0)}

        for action in gameState.getLegalActions(0):
            actionVal[action] = alpha_beta(self, gameState.generateSuccessor(0, action),
                                                                  1, 0, float("-inf"), float('inf'))
        return list(sorted(actionVal.items(), key=lambda x: x[1], reverse=True))[0][0]

        util.raiseNotDefined()


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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function.
    """
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
