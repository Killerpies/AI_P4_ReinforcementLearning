"""
analysis.py
-----------
Licensing Information:  You are free to use or extend these projects for
educational purposes provided that (1) you do not distribute or publish
solutions, (2) you retain this notice, and (3) you provide clear
attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

Attribution Information: The Pacman AI projects were developed at UC Berkeley.
The core projects and autograders were primarily created by John DeNero
(denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
Student side autograding was added by Brad Miller, Nick Hay, and
Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #its
######################

Set the given parameters to obtain the specified policies through
value iteration.
"""


def question2():
    answer_discount = 0.9
    # no noise definitely makes it easier
    answer_noise = 0
    return answer_discount, answer_noise


def question3a():
    answer_discount = None
    answer_noise = None
    answer_living_reward = None
    return answer_discount, answer_noise, answer_living_reward
    # If not possible, return 'NOT POSSIBLE'


def question3b():
    answer_discount = None
    answer_noise = None
    answer_living_reward = None
    return answer_discount, answer_noise, answer_living_reward
    # If not possible, return 'NOT POSSIBLE'


def question3c():
    answer_discount = None
    answer_noise = None
    answer_living_reward = None
    return answer_discount, answer_noise, answer_living_reward
    # If not possible, return 'NOT POSSIBLE'


def question3d():
    answer_discount = None
    answer_noise = None
    answer_living_reward = None
    return answer_discount, answer_noise, answer_living_reward
    # If not possible, return 'NOT POSSIBLE'


def question3e():
    answer_discount = None
    answer_noise = None
    answer_living_reward = None
    return answer_discount, answer_noise, answer_living_reward
    # If not possible, return 'NOT POSSIBLE'


def question6():
    answer_epsilon = None
    answer_learning_rate = None
    return answer_epsilon, answer_learning_rate


if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
