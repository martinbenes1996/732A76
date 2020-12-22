# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 12:39:47 2020

@author: martin
"""

import numpy as np
from typing import Any,Union

class Container:
    """Overflow container. Holds only N best (lowest) items by score.
    
    Attributes:
        _N (int): Container size.
        _k (int): Current number of items in container.
        _data (list): Container contents, ordered (desc) by scores
        _score (np.array): Numpy array of scores, indices correspond with data.
                           Based on this the decision when adding is done.
    """
    def __init__(self, N:int, minimize = False):
        """Constructor. Sets the fixed size.
        
        Args:
            N (int): Container size
            minimize (bool): True for minimizing, False for maximizing.
        """
        if N <= 0: raise ValueError("N must be positive")
        
        # initialize
        self._N, self._k = int(N), 0
        self._minimize = minimize
        self._data = [0 for _ in range(self._N)]
        init_val = "-inf" if self._minimize else "inf"
        self._score = np.array([float(init_val) for _ in range(self._N)])
        
    def add(self, item: Any, score: float) -> None:
        """Adds the item to the container with regard to score.
        
        Based on score the position is decided or the item is ignored.
        Args:
            item (any): Data.
            score (float): Score.
        """
        if not self._minimize: score = -score
        
        # greater than all - ignore
        if all(self._score < score): return
        
        # find position
        i = np.argmax(self._score > score)
        # make space
        self._data[(i+1):(self._N)] = self._data[(i):(self._N-1)]
        self._score[(i+1):(self._N)] = self._score[(i):(self._N-1)]
        # insert
        if not self._minimize: score = -score
        self._data[i] = item
        self._score[i] = score
        # increment counter
        if self._k < self._N:
            self._k += 1
            
    def __getitem__(self, i:int) -> tuple:
        # negative indexing
        if i < 0:
            i = self._k + i

        # positive indexing - touching behind
        if i < 0 or self._k <= i:
            raise IndexError("out of range")
            
        return (self._data[i],self._score[i])
    
    def get(self) -> Any:
        return self._data
    def score(self) -> float:
        return self._score
    def to_numpy(self, score = True) -> Union[tuple,np.array]:
        if score:
            return np.array(self.get()), np.array(self.score())
        else:
            return np.array(self.get())
    