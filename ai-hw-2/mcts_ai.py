#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""
MCTS AI player for Othello.
"""

import random
import numpy as np
from six.moves import input
from othello_shared import get_possible_moves, play_move, compute_utility


class Node:
    def __init__(self, state, player, parent, children, v=0, N=0):
        self.state = state
        self.player = player
        self.parent = parent
        self.children = children
        self.value = v
        self.N = N

    def get_child(self, state):
        for c in self.children:
            if (state == c.state).all():
                return c
        return None


def select(root, alpha):
    """ Starting from given node, find a terminal node or node with unexpanded children.
    If all children of a node are in tree, move to the one with the highest UCT value.

    Args:
        root (Node): MCTS tree root node
        alpha (float): Weight of exploration term in UCT

    Returns:
        node (Node): Node at bottom of MCTS tree

    select() takes in the root node and alpha value for UCT calculation. If the current node's
    children list contains all possible successors of the board state, then repeatedly move to
    the one with the highest UCT value. You can obtain these successors using the imported
    get possible moves() and play move() functions. If the current node has at least one
    successor not in the tree or is a terminal state, then it stops and returns the current node.
    """
    possible_moves = get_possible_moves(root.state, root.player)
    children = root.children
    if possible_moves and children and len(possible_moves) == len(children):
        c = max(children, key=lambda x: x.value + alpha * np.sqrt(2 * np.log(root.N) / x.N))
        return select(c, alpha)
    else:
        return root


def expand(node):
    """ Add a child node of state into the tree if it's not terminal.

    Args:
        node (Node): Node to expand

    Returns:
        leaf (Node): Newly created node (or given Node if already leaf)

    expand() attempts to expand the tree. It finds a successor of the given state that is currently
    not in node.children, creates a new leaf node, and adds it node.children. It then returns
    the leaf node. If node has no successors, then it simply returns node back.
    """
    possible_moves = get_possible_moves(node.state, node.player)
    children = node.children
    if possible_moves != [] and len(possible_moves) > len(children):
        for m in possible_moves:
            s = play_move(node.state, node.player, m[0], m[1])
            if node.get_child(s) is None:
                new = Node(s, 3 - node.player, node, [], 0, 0)
                node.children.append(new)
                return new
    return node

def simulate(node):
    """ Run one game rollout using from state to a terminal state.
    Use random playout policy.

    Args:
        node (Node): Leaf node from which to start rollout.

    Returns:
        utility (int): Utility of final state

    simulate() runs a rollout starting from the given node. One way to do so is to simply
    execute a random move at each node until reaching a terminal state. It then computes and
    returns the utility of the final state (you can use compute utility()).
    """
    state = node.state
    player = node.player
    while True:
        moves = get_possible_moves(state, player)
        if moves:
            move = random.choice(moves)
            state = play_move(state, player, move[0], move[1])
        else:
            break
        player = 3 - player
    return compute_utility(state)

def backprop(node, utility):
    """ Backpropagate result from state up to the root.
    Every node has N, number of plays, incremented
    If node's parent is dark (1), then node's value increases
    Otherwise, node's value decreases.

    Args:
        node (Node): Leaf node from which rollout started.
        utility (int): Utility of simulated rollout.

    backprop() backpropagates the computed utility from node back up to the root. First, we
    increment the current node's N value. Next, the node's value update depends on the player's
    utility. For the light player (2) we can use utility directly, since their parent (1) wants to
    maximize these values. For the dark player (1), we need to use the negative of utility for
    the opposite reason. The new (average) value of each node can then be computed as follows:
    node.value = (node.value * (node.N - 1) + player_utility) / node.N
    """
    while node is not None:
        node.N += 1
        if node.player == 1:
            node.value = (node.value * (node.N - 1) - utility) / node.N
        else:
            node.value = (node.value * (node.N - 1) + utility) / node.N
        node = node.parent
    return


def mcts(state, player, rollouts=100, alpha=5):
    # MCTS main loop: Execute four steps rollouts number of times
    # Then return successor with highest number of rollouts
    root = Node(state, player, None, [], 0, 1)
    for i in range(rollouts):
        leaf = select(root, alpha)
        new = expand(leaf)
        utility = simulate(new)
        backprop(new, utility)

    move = None
    plays = 0
    for m in get_possible_moves(state, player):
        s = play_move(state, player, m[0], m[1])
        if root.get_child(s).N > plays:
            plays = root.get_child(s).N
            move = m

    return move


####################################################
def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("MCTS AI")        # First line is the name of this AI
    color = int(input())    # 1 for dark (first), 2 for light (second)

    while True:
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        next_input = input()
        status, dark_score_s, light_score_s = next_input.strip().split()

        if status == "FINAL":
            print()
        else:
            board = np.array(eval(input()))
            movei, movej = mcts(board, color)
            print("{} {}".format(movei, movej))


if __name__ == "__main__":
    run_ai()