"""
q_learning_agents.py
------------------
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
from feature_extractors import *
from game import *
from learning_agents import ReinforcementAgent
from util import random

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - compute_value_from_q_values
        - compute_action_from_q_values
        - get_q_value
        - get_action
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.get_legal_actions(state)
          which return_val legal actions for a state
    """

    def __init__(self, **args):
        """
        You can initialize Q-values here...
        :param args:
        """
        ReinforcementAgent.__init__(self, **args)
        # *** YOUR CODE HERE ***
        self.values = util.Counter()

    def get_q_value(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """

        # *** YOUR CODE HERE ***

        return self.values[(state,action)]

    def compute_value_from_q_values(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """

        # *** YOUR CODE HERE ***

        actions = self.get_legal_actions(state)
        if len(actions) == 0:
            return 0.0
        # compute_action_from_q_values(state)
        bestAction = self.get_policy(state)
        return self.get_q_value(state, bestAction)

    def compute_action_from_q_values(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.line
        """

        # *** YOUR CODE HERE ***

        legalActions = self.get_legal_actions(state)
        # is_terminal
        if len(legalActions) == 0:
            return None

        # dict to store actions
        # no util.counter because of this:
        # Beware of the argmax function from util.Counter!
        values = {}
        bestQValue = float('-inf')
        for action in legalActions:
            # get que val for state action
            tempQVal = self.get_q_value(state, action)
            # insert into dict
            values[action] = tempQVal
            # looking for biggest qVal
            if tempQVal > bestQValue:
                bestQValue = tempQVal

        bestActions = []
        for action, value in values.items():
            # List of all the highest qvalue actions so we can break tie
            if value == bestQValue:
                bestActions.append(action)
        return random.choice(bestActions)

    def get_action(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flip_coin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legal_actions = self.get_legal_actions(state)
        action = None

        # *** YOUR CODE HERE ***

        util.raise_not_defined()


    def update(self, state, action, next_state, reward):
        """
          The parent class calls this to observe a
          state = action => next_state and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """

        # *** YOUR CODE HERE ***

        currentQVal = self.get_q_value(state, action)
        nextStateQValue = self.get_value(next_state)
        # currentval + learningRate * (reward + (discount * nextStateVal) - currentVal)
        newQValue = currentQVal + self.alpha * (reward + (self.discount * nextStateQValue) - currentQVal)
        # change this state/action qvalue to the new q value
        self.values[(state, action)] = newQValue

    def get_policy(self, state):
        return self.compute_action_from_q_values(state)

    def get_value(self, state):
        return self.compute_value_from_q_values(state)


class PacmanQAgent(QLearningAgent):
    """Exactly the same as QLearningAgent, but with different default parameters"""

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, num_training=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['num_training'] = num_training
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def get_action(self, state):
        """
        Simply calls the get_action method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.get_action(self, state)
        self.do_action(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite get_q_value
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.feat_extractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def get_weights(self):
        return self.weights

    def get_q_value(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """

        # *** YOUR CODE HERE ***

        util.raise_not_defined()

    def update(self, state, action, next_state, reward):
        """
           Should update your weights based on transition
        """

        # *** YOUR CODE HERE ***

        util.raise_not_defined()

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodes_so_far == self.num_training:
            # you might want to print your weights here for debugging

            # *** YOUR CODE HERE ***

            pass
