# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 13:09:52 2020

@author: martin
"""

import sys
import unittest

sys.path.append(".")
import ovfw

class Test_Container(unittest.TestCase):
    
    def test_invalid_N(self):
        # N negative
        try: c = ovfw.Container(N = -7)
        except: raised = True
        else: raised = False
        self.assertTrue(raised)
        
        # N = 0
        try: c = ovfw.Container(N = 0)
        except: raised = True
        else: raised = False
        self.assertTrue(raised)
        
        # N not a number
        try: c = ovfw.Container(N = "a number")
        except: raised = True
        else: raised = False
        self.assertTrue(raised)
    
    def test_indexing(self):
        # instantiate
        c = ovfw.Container(3)
        
        # fill the container
        c.add(1,10)
        c.add(3,20)
        c.add(-1,30)
        
        # index
        i0 = c[0]
        self.assertAlmostEqual(i0[0], 1)
        self.assertAlmostEqual(i0[1], 10)
        i1 = c[1]
        self.assertAlmostEqual(i1[0], 3)
        self.assertAlmostEqual(i1[1], 20)
        i2 = c[2]
        self.assertAlmostEqual(i2[0], -1)
        self.assertAlmostEqual(i2[1], 30)
    
    def test_out_of_range(self):
        # instantiate
        c = ovfw.Container(2)
        
        # fill part of container
        c.add(1,10)
        
        # positive indexing
        i0 = c[0]
        self.assertAlmostEqual(i0[0], 1)
        self.assertAlmostEqual(i0[1], 10)
        
        # negative indexing
        i0 = c[-1]
        self.assertAlmostEqual(i0[0], 1)
        self.assertAlmostEqual(i0[1], 10)
        
        # positive indexing out of range
        try:
            i1 = c[1]
        except IndexError: raised = True
        else: raised = False
        self.assertTrue(raised)
        
        # negative indexing out of range
        try:
            i_1 = c[-2]
        except IndexError: raised = True
        else: raised = False
        self.assertTrue(raised)
    
    def test_add(self):
        b = ovfw.Container(3)
        
        # first element (index 0)
        b.add(5,1)
        self.assertAlmostEqual(b._data[0], 5)
        self.assertAlmostEqual(b._score[0], 1)
        
        # lower element (index 0)
        b.add(6,.9)
        self.assertAlmostEqual(b._data[0], 6)
        self.assertAlmostEqual(b._score[0], .9)
        self.assertAlmostEqual(b._data[1], 5)
        self.assertAlmostEqual(b._score[1], 1)
        
        # element in between (index 1)
        b.add(7,.95)
        self.assertAlmostEqual(b._data[0], 6)
        self.assertAlmostEqual(b._score[0], .9)
        self.assertAlmostEqual(b._data[1], 7)
        self.assertAlmostEqual(b._score[1], .95)
        self.assertAlmostEqual(b._data[2], 5)
        self.assertAlmostEqual(b._score[2], 1)
        
        # new first element (index 0)
        b.add(8,.89)
        self.assertAlmostEqual(b._data[0], 8)
        self.assertAlmostEqual(b._score[0], .89)
        self.assertAlmostEqual(b._data[1], 6)
        self.assertAlmostEqual(b._score[1], .9)
        self.assertAlmostEqual(b._data[2], 7)
        self.assertAlmostEqual(b._score[2], .95)
        
        # new last element (index 2)
        b.add(9,.925)
        self.assertAlmostEqual(b._data[0], 8)
        self.assertAlmostEqual(b._score[0], .89)
        self.assertAlmostEqual(b._data[1], 6)
        self.assertAlmostEqual(b._score[1], .9)
        self.assertAlmostEqual(b._data[2], 9)
        self.assertAlmostEqual(b._score[2], .925)
        
        # low element (ignored)
        b.add(10,.93)
        self.assertAlmostEqual(b._data[0], 8)
        self.assertAlmostEqual(b._score[0], .89)
        self.assertAlmostEqual(b._data[1], 6)
        self.assertAlmostEqual(b._score[1], .9)
        self.assertAlmostEqual(b._data[2], 9)
        self.assertAlmostEqual(b._score[2], .925)

# logging
if __name__ == "__main__":
    import logging
    logging.basicConfig(level = logging.WARNING)

# run unittests
if __name__ == "__main__":
    unittest.main()
    
    