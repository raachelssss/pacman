# valueIterationAgents.py
# -----------------------
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

# valueIterationAgents.py
# -----------------------
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

import numpy as np

import gridworld
import mdp, util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
          Run the value iteration algorithm. Note that in standard
          value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        allStates = mdp.getStates()
        k_grid = {states:0 for states in allStates}

        # width = max([i for i,j in allStates]) + 1
        # height = max([j for i,j in allStates]) + 1
        # # k_grid = [[0]*width for j in range(height)]
        # print(k_grid)
        # k1_grid = [[0]*width for j in range(height)]
        # print('values:, ', self.values) #{'TERMINAL_STATE': 0, (0, 0): 0} -- keys: state, values: value
        #iteration should update computed values to self.values[state]
        for k in range(self.iterations):
            discount = self.discount
            for state in k_grid:
                if state == 'TERMINAL_STATE':
                    k_grid[state] = 0
                    # self.values[state] = 0
                    continue
                actions = mdp.getPossibleActions(state)
                candidates = []
                for a in actions: #slow, fast
                    T = mdp.getTransitionStatesAndProbs(state, a) # ( (i',j'), prob )
                    sum = 0
                    for nextState, prob in T: #cool, warm
                        reward = mdp.getReward(state, a, nextState)
                        v = self.values[nextState]
                        sum += prob * (reward + discount * v)
                    candidates.append(sum)
                vStar = max(candidates)
                k_grid[state] = vStar

            for state in k_grid:
                self.values[state] = k_grid[state]


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        T = mdp.getTransitionStatesAndProbs(state, action)
        discount = self.discount
        q = 0
        for nextState, prob in T:

        # print('T:, ', T) #[('TERMINAL_STATE', 1.0)] -- (state, prob)
        # print('state:, ', state, 'action:, ' , action) #(0, 0) exit
        # print('values:, ', self.values) #{'TERMINAL_STATE': 0, (0, 0): 0} -- keys: state, values: value
        # print('possible actions:, ', mdp.getPossibleActions(state))
            reward = mdp.getReward(state, action, nextState)
            value = self.getValue(nextState)
            q += prob *(reward + discount*value)

        return q
        util.raiseNotDefined()


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the= values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        mdp = self.mdp
        actions = mdp.getPossibleActions(state)
        if state == 'TERMINAL_STATE' or len(actions) == 0:
            return None
        legalActions = {a:0 for a in actions}
        for a in actions:
            legalActions[a] = self.computeQValueFromValues(state, a)
        return max(legalActions, key=legalActions.get)


    # def computeActionFromValues(self, state):
    #     """
    #       The policy is the best action in the given state
    #       according to the values currently stored in self.values.
    #
    #       You may break ties any way you see fit.  Note that if
    #       there are no legal actions, which is the case at the
    #       terminal state, you should return None.
    #     """
    #
    #     if self.mdp.isTerminal(state):
    #         return None
    #     legalActions = self.mdp.getPossibleActions(state)
    #     actions = []
    #     for action in legalActions:
    #         actions.append(self.computeQValueFromValues(state, action))
    #     best = np.argmax(actions)
    #     return legalActions[best]


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
