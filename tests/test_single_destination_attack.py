'''
Tests for the single_destination_attack policy
'''

import unittest
import MAS_network as MAS
from utils import generate_uniform, generate_asymmetric, is_equal
from single_destination_attack import SingleDestinationAttack
import numpy as np


__author__ = 'jeromethai'


class TestSingleDestinationAttack(unittest.TestCase):
    

    def test_single_destination_attack(self):
        # generate asymmetric MAS_network with availabilities [0.5  0.5  1.0]
        # and attack on station 2 with budget = 1.0 (default)
        network = MAS.Network(*generate_asymmetric())
        attack_rates, attack_routing = SingleDestinationAttack(network, 2).apply()
        network.update(attack_rates, attack_routing)
        self.assertTrue(is_equal(network.new_availabilities(), np.array([1./3, 1./3, 1.])))
        # now attack on station 1
        attack_rates, attack_routing = SingleDestinationAttack(network, 1).apply()
        network.update(attack_rates, attack_routing)
        self.assertTrue(is_equal(network.new_availabilities(), np.array([1./3, 1., 2./3])))
        # now attack on station 1 with 0.49 budget -> inefficient attack
        network.budget = 0.49
        attack_rates, attack_routing = SingleDestinationAttack(network, 1).apply()
        network.update(attack_rates, attack_routing)
        self.assertTrue(is_equal(network.new_availabilities(), np.array([.5, .5, 1.])))        

