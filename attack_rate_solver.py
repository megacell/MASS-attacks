'''
Attack routing solver:
Optimizing for the Optimal Attack problem with the attack routing probabilities fixed
'''


import numpy as np

__author__ = 'jeromethai'



class AttackRateSolver:
    def __init__(self, network, attack_routing, k, nu):
        # Class for the Attack Rate Solver
        self.network = network
        self.attack_routing = attack_routing
        # index k such that a_k is set to 1
        self.k = k


    def gradient_computation(self):
        # compute the gradient of the objective with respect to the attack rates


    
