"""
COMS W4701 Artificial Intelligence - Programming Homework 3

A dynamic programming agent for a stochastic task environment
"""

import random
import math
import sys


class DP_Agent(object):

    def __init__(self, states, parameters):
        self.gamma = parameters["gamma"]
        self.V0 = parameters["V0"]

        self.states = states
        self.values = {}
        self.policy = {}

        for state in states:
            self.values[state] = parameters["V0"]
            self.policy[state] = None


    def setEpsilon(self, epsilon):
        pass

    def setDiscount(self, gamma):
        self.gamma = gamma

    def setLearningRate(self, alpha):
        pass


    def choose_action(self, state, valid_actions):
        return self.policy[state]

    def update(self, state, action, reward, successor, valid_actions):
        pass


    def value_iteration(self, valid_actions, transition):
        """ Computes all optimal values using value iteration and stores them in self.values.

        Args:
            valid_actions (Callable): Function that returns a list of actions given a state.
            transition (Callable): Function that returns successor state and reward given a state and action.
        """
        epsilon = 0.01
        delta = 1
        while delta > epsilon:
            delta = 0
            for state in self.states:
                v = self.values[state]
                max_v = -sys.maxsize
                for action in valid_actions(state):
                    s_prime, r = transition(state, action)
                    max_v = max(max_v, r + self.gamma * self.values[s_prime])
                self.values[state] = max_v
                delta = max(delta, abs(v - self.values[state]))
        pass


    def policy_extraction(self, valid_actions, transition):
        """ Computes all optimal actions using value iteration and stores them in self.policy.

        Args:
            valid_actions (Callable): Function that returns a list of actions given a state.
            transition (Callable): Function that returns successor state and reward given a state and action.
        """
        for state in self.states:
            max_v = -sys.maxsize
            for action in valid_actions(state):
                s_prime, r = transition(state, action)
                if r + self.gamma * self.values[s_prime] > max_v:
                    self.policy[state] = action
                    max_v = r + self.gamma * self.values[s_prime]
        pass
