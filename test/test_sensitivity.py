# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
from neet.sensitivity import *
from neet.landscape import StateSpace
import neet.boolean as NB

class TestSensitivityWTNetwork(unittest.TestCase):
    class IsBooleanNetwork(object):
        def update(self, lattice):
            pass
        def state_space(self):
            return StateSpace(1)

    class IsNotBooleanNetwork(object):
        def update(self, lattice):
            pass
        def state_space(self):
            return StateSpace([2,3,4])

    class IsNotNetwork(object):
        pass

    def test_sensitivity_net_type(self):
        with self.assertRaises(TypeError):
            sensitivity(self.IsNotNetwork(), [0,0,0])

        with self.assertRaises(TypeError):
            sensitivity(self.IsNotBooleanNetwork(), [0,0,0])

    def test_hamming_neighbors_input(self):
        with self.assertRaises(ValueError):
            hamming_neighbors([0,1,2])
        
        with self.assertRaises(ValueError):
            hamming_neighbors([[0,0,1],[1,0,0]])

    def test_hamming_neighbors_example(self):
        state = [0,1,1,0]
        neighbors = [[1,1,1,0],
                     [0,0,1,0],
                     [0,1,0,0],
                     [0,1,1,1]]
        self.assertTrue(np.array_equal(neighbors,hamming_neighbors(state)))

    def test_sensitivity(self):
        net = NB.WTNetwork([[1,-1],[0,1]],[0.5,0])
        self.assertEqual(2,sensitivity(net,[0,0]))

    def test_average_sensitivity_net_type(self):
        with self.assertRaises(TypeError):
            average_sensitivity(self.IsNotNetwork())

        with self.assertRaises(TypeError):
            average_sensitivity(self.IsNotBooleanNetwork())

    def test_average_sensitivity_lengths(self):
        net = NB.WTNetwork([[1,-1],[0,1]],[0.5,0])
    
        with self.assertRaises(ValueError):
            average_sensitivity(net,states=[[0,0],[0,1]],weights=[0,1,2])

    def test_average_sensitivity(self):
        net = NB.WTNetwork([[1,-1],[0,1]],[0.5,0])
        self.assertEqual(1.5,average_sensitivity(net))

