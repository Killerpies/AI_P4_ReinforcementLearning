"""
value_iteration_agents.py
-----------------------
Licensing Information:  You are free to use or extend these projects for
educational purposes provided that (1) you do not distribute or publish
solutions, (2) you retain this notice, and (3) you provide clear
attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

Attribution Information: The Pacman AI projects were developed at UC Berkeley.
The core projects and autograders were primarily created by John DeNero
(denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
Student side autograding was added by Brad Miller, Nick Hay, and
Pieter Abbeel (pabbeel@cs.berkeley.edu).
"""

import util

from learning_agents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learning_agents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.get_states()
              mdp.get_possible_actions(state)
              mdp.get_transition_states_and_probs(state, action)
              mdp.get_reward(state, action, next_state)
              mdp.is_terminalCounter((state)
        """
        super().__init__()
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.valueGetter()
        # Write value iteration code here

        # *** YOUR CODE HERE ***
    def valueGetter(self):
        for count in range(0, self.iterations):
            # make copy
            newValues = self.values.copy()
            # loop states
            for state in self.mdp.get_states():
                # skip if terminal
                if self.mdp.is_terminal(state):
                    continue
                # use functions we already made for value and action
                bestAction = self.compute_action_from_values(state)
                qValue = self.compute_q_value_from_values(state, bestAction)
                # add to the list of new values
                newValues[state] = qValue
            # new values replace old values
            self.values = newValues

    def get_value(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def compute_q_value_from_values(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """

        # *** YOUR CODE HERE ***
        nextProbsAndTransitions = self.mdp.get_transition_states_and_probs(state, action)
        value = 0

        for nextState, probs in nextProbsAndTransitions:
            # get next state reward Value
            nextRewardValue = self.mdp.get_reward(state, action, nextState)
            # q value
            qvalue = self.values[nextState]
            # quickmath - probability * (reward + discount * qvalue)
            value += probs * (nextRewardValue + self.discount * qvalue)
        return value

    def compute_action_from_values(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # print(self.values)
        # mdp.get_states()
        # self.mdp.get_possible_actions(state)
        # mdp.get_transition_states_and_probs(state, action)
        # mdp.get_reward(state, action, next_state)
        # mdp.is_terminal(state)
        # *** YOUR CODE HERE ***

        if self.mdp.is_terminal(state):
            return None

        actions = self.mdp.get_possible_actions(state)
        values = util.Counter()
        # for each action use save action, value to dict
        for action in actions:
            values[action] = self.compute_q_value_from_values(state, action)
        # biggest action value is best value
        bestAction = values.arg_max()
        return bestAction


    def get_policy(self, state):
        return self.compute_action_from_values(state)

    def get_action(self, state):
        """Returns the policy at the state (no exploration)."""
        return self.compute_action_from_values(state)

    def get_q_value(self, state, action):
        return self.compute_q_value_from_values(state, action)
