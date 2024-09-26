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
        # Collect legal moves and successor states
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
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        # 1. nearest food
        foodList = newFood.asList() 
        if foodList:
            closestFoodDist = min([util.manhattanDistance(newPos, food) for food in foodList])
        else:
            closestFoodDist = 0 

        # 2. nearest ghost
        ghostDistances = []
        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostPos = ghostState.getPosition()
            distance = util.manhattanDistance(newPos, ghostPos)

            if scaredTime > 0:
            # ghost at "scared" state, better to chase them
                ghostDistances.append(-distance)  
            else:
            # should keep away from the ghost
                ghostDistances.append(distance)

        closestGhostDist = min(ghostDistances) if ghostDistances else math.inf

        # 3. number of remained food
        remainingFoodCount = successorGameState.getNumFood()

        # 4. total score
        score = successorGameState.getScore()

        # calculate final score (weights need to adjusted)
        evaluation = score
        evaluation -= 1.5 * closestFoodDist  # be close to the food 
        evaluation += 1.0 * closestGhostDist  # be far away from the ghost or be close to the ghost when it is "scared"
        evaluation -= 50 * remainingFoodCount - currentGameState.getNumFood()  # less food remain
        if action == Directions.STOP:  # prevent stop too long
            evaluation -= 1000

        return evaluation


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
        action, _ = self.minimax(gameState, 0, 0)
        return action
    
    def minimax(self, gameState: GameState, current_agent: int, current_depth: int):

        if current_depth == self.depth or gameState.isWin() or gameState.isLose():
            return None, self.evaluationFunction(gameState)

        # Pacman
        if current_agent == 0:
            return self.max_value(gameState, current_depth)
        # Ghost
        else:
            return self.min_value(gameState, current_agent, current_depth)

    def max_value(self, gameState: GameState, current_depth: int):
        max_score = float('-inf')
        best_action = None
        # find out the best action with highest score
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            _, score = self.minimax(successor, 1, current_depth)
            if score > max_score:
                max_score = score
                best_action = action
        return best_action, max_score

    def min_value(self, gameState: GameState, current_agent: int, current_depth: int):

        min_score = float('inf')
        best_action = None
        next_agent = (current_agent + 1) % gameState.getNumAgents() 
        next_depth = current_depth + 1 if next_agent == 0 else current_depth  #update depth, if the next agent is Pacmen

        # find out the best action with the lowest score
        for action in gameState.getLegalActions(current_agent):
            successor = gameState.generateSuccessor(current_agent, action)
            _, score = self.minimax(successor, next_agent, next_depth)
            if score < min_score:
                min_score = score
                best_action = action
        return best_action, min_score 
    
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        action, _ = self.minimax(gameState, 0, 0, float('-inf'), float('inf'))  # Initialize alpha and beta
        return action
    
    def minimax(self, gameState: GameState, current_agent: int, current_depth: int, alpha: float, beta: float):
        # Terminal state (win/lose or depth reached)
        if current_depth == self.depth or gameState.isWin() or gameState.isLose():
            return None, self.evaluationFunction(gameState)

        # Pacman (Maximizing player)
        if current_agent == 0:
            return self.max_value(gameState, current_depth, alpha, beta)

        # Ghost (Minimizing player)
        else:
            return self.min_value(gameState, current_agent, current_depth, alpha, beta)

    def max_value(self, gameState: GameState, current_depth: int, alpha: float, beta: float):
        max_score = float('-inf')
        best_action = None

        # Explore actions for Pacman (agent 0)
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            _, score = self.minimax(successor, 1, current_depth, alpha, beta)
            
            if score > max_score:
                max_score = score
                best_action = action

            # Update alpha (best option for max)
            alpha = max(alpha, max_score)

            # Alpha-beta pruning
            if max_score > beta:
                break  # Prune this branch
        
        return best_action, max_score

    def min_value(self, gameState: GameState, current_agent: int, current_depth: int, alpha: float, beta: float):
        min_score = float('inf')
        best_action = None
        next_agent = (current_agent + 1) % gameState.getNumAgents() 
        next_depth = current_depth + 1 if next_agent == 0 else current_depth  # Update depth if it's Pacman's turn next

        # Explore actions for the ghost (agent > 0)
        for action in gameState.getLegalActions(current_agent):
            successor = gameState.generateSuccessor(current_agent, action)
            _, score = self.minimax(successor, next_agent, next_depth, alpha, beta)

            if score < min_score:
                min_score = score
                best_action = action

            # Update beta (best option for min)
            beta = min(beta, min_score)

            # Alpha-beta pruning
            if min_score < alpha:
                break  # Prune this branch
        
        return best_action, min_score

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
        action, _ = self.expectimax(gameState, 0, 0)
        return action
    
    def expectimax(self, gameState: GameState, current_agent: int, current_depth: int):

        if current_depth == self.depth or gameState.isWin() or gameState.isLose():
            return None, self.evaluationFunction(gameState)

        # Pacman
        if current_agent == 0:
            return self.max_value(gameState, current_depth)
        # Ghost
        else:
            return self.expect_value(gameState, current_agent, current_depth)

    def max_value(self, gameState: GameState, current_depth: int):
        max_score = float('-inf')
        best_action = None
        # find out the best action with highest score
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            _, score = self.expectimax(successor, 1, current_depth)
            if score > max_score:
                max_score = score
                best_action = action
        return best_action, max_score

    def expect_value(self, gameState: GameState, current_agent: int, current_depth: int):

        expect_score = 0
        random_action = random.randint(0, len(gameState.getLegalActions()))
        next_agent = (current_agent + 1) % gameState.getNumAgents() 
        next_depth = current_depth + 1 if next_agent == 0 else current_depth  #update depth, if the next agent is Pacmen

        # find out the best action with the lowest score
        for action in gameState.getLegalActions(current_agent):
            successor = gameState.generateSuccessor(current_agent, action)
            _, score = self.expectimax(successor, next_agent, next_depth)
            expect_score = expect_score + score / len(gameState.getLegalActions())
        return random_action, expect_score

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Get useful information from the current game state
    pacmanPos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    # Get the current game score
    score = currentGameState.getScore()

    # 1. Distance to the nearest food
    foodList = foodGrid.asList()
    if foodList:
        closestFoodDist = min([manhattanDistance(pacmanPos, food) for food in foodList])
    else:
        closestFoodDist = 0  # No food left

    # 2. Distance to the nearest ghost
    ghostDistances = []
    scaredGhostDistances = []
    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        distance = manhattanDistance(pacmanPos, ghostPos)

        if ghostState.scaredTimer > 0:
            scaredGhostDistances.append(distance)  # Add scared ghost distance
        else:
            ghostDistances.append(distance)  # Add non-scared ghost distance

    # 3. Number of food pellets left
    remainingFoodCount = currentGameState.getNumFood()

    # 4. Distance to the nearest capsule (power pellet)
    if capsules:
        closestCapsuleDist = min([manhattanDistance(pacmanPos, capsule) for capsule in capsules])
    else:
        closestCapsuleDist = float('inf')  # No capsules left

    # 5. Evaluate the state based on the gathered information
    evaluation = score

    # Encourage Pacman to go closer to food
    if closestFoodDist:
        evaluation += 10.0 / closestFoodDist  # The closer to food, the better

    # Penalize Pacman if it's too close to non-scared ghosts
    if ghostDistances:
        closestGhostDist = min(ghostDistances)
        if closestGhostDist > 0:
            evaluation -= 5.0 / closestGhostDist  # The closer to ghosts, the worse

    # Encourage Pacman to chase scared ghosts
    if scaredGhostDistances:
        closestScaredGhostDist = min(scaredGhostDistances)
        evaluation += 20.0 / closestScaredGhostDist  # The closer to scared ghosts, the better

    # Encourage Pacman to go towards capsules
    if closestCapsuleDist < float('inf'):
        evaluation += 15.0 / closestCapsuleDist  # The closer to a capsule, the better

    # Penalize having more remaining food and capsules
    evaluation -= 4 * remainingFoodCount  # The fewer food pellets, the better
    evaluation -= 20 * len(capsules)  # The fewer capsules, the better

    # Return the evaluation score
    return evaluation
better = betterEvaluationFunction
