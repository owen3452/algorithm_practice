#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zhen
# @email    : zhendch@qq.com

import numpy as np
from DecisionTree import DecisionTree


class AdaboostingClassifier(object):
    '''
    n_estimators: 迭代次数
    criterion: gini,entropy
    type: distrete, continuous
    splitter: best, random
    max_depth: 最大深度
    min_impurity_decrease： 熵or基尼系数减少的阈值，小于阈值不分裂
    min_impurity_split： 节点熵or基尼系数的阈值，小于阈值不分裂
    min_samples_split： 节点样本数的阈值，小于阈值不分裂
    label: 1, -1
    '''
    def __init__(self, n_estimators=0, criterion='gini', max_depth=None, type='distrete', splitter='best',
                 min_samples_split=2, min_impurity_decrease=0.0, min_impurity_split=None):
        self._n_estimators = n_estimators
        self._criterion = criterion
        self._max_depth = max_depth
        self._type = type
        self._splitter = splitter
        self._min_samples_split = min_samples_split
        self._min_impurity_decrease = min_impurity_decrease
        self._min_impurity_split = min_impurity_split

    def fit(self, X, y):
        m, n = np.shape(X)
        for i in range(self._n_estimators):
            tree_tmp = DecisionTree(type=self._type, criterion=self._criterion, splitter=self._splitter,
                                    min_impurity_decrease=self._min_impurity_decrease,
                                    min_impurity_split=self._min_impurity_split,
                                    min_samples_split=self._min_samples_split, max_depth=self._max_depth)
            tree_tmp.fit(X, y)
            predict_tmp = tree_tmp.predict(X)