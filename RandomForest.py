#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zhen
# @email    : zhendch@qq.com

import numpy as np
import random
from collections import Counter
from DecisionTree import DecisionTree

class RandomForestClassifier(object):
    '''
    n_estimators: number of trees in the forest
    criterion: gini,entropy
    type: distrete, continuous
    splitter: best, random
    max_depth: 最大深度
    min_impurity_decrease： 熵or基尼系数减少的阈值，小于阈值不分裂
    min_impurity_split： 节点熵or基尼系数的阈值，小于阈值不分裂
    min_samples_split： 节点样本数的阈值，小于阈值不分裂
    '''
    def __init__(self, n_estimators=10, criterion='gini', max_depth=None, type='distrete', splitter='best',
                 min_samples_split=2,  min_impurity_decrease=0.0, min_impurity_split=None,
                 bagging_fraction=1.0, feature_fraction=1.0):
        self._n_estimators = n_estimators
        self._criterion = criterion
        self._max_depth = max_depth
        self._type = type
        self._splitter = splitter
        self._min_samples_split = min_samples_split
        self._min_impurity_decrease = min_impurity_decrease
        self._min_impurity_split = min_impurity_split
        self._bagging_fraction = bagging_fraction
        self._feature_fraction = feature_fraction

    def random_sample(self, X, y, m, n):
        # X随机取m行n列数据 y取相应行
        row, column = np.shape(X)
        row = random.sample(range(row), m)
        column = random.sample(range(column), n)
        return X[row][:,column], y[row], row, column

    def fit(self, X, y):
        m, n = np.shape(X)
        self._tree_list = []
        for i in range(self._max_depth):
            tree_tmp = DecisionTree(type=self._type, criterion=self._criterion, splitter=self._splitter,
                                    min_impurity_decrease=self._min_impurity_decrease,
                                    min_impurity_split=self._min_impurity_split,
                                    min_samples_split=self._min_samples_split, max_depth=self._max_depth)
            X_train, y_train, row, column = self.random_sample(X, y, self._bagging_fraction*m, self._feature_fraction*n)
            tree_tmp.fit(X_train, y_train)
            self._tree_list.append([column, tree_tmp])

    def predict_prob(self, X):
        result = []
        for x in X:
            prob_list = []
            for i in self._tree_list:
                predict_tree = i[1].predict(x[:,i[0]])
                prob_list.append(predict_tree)
            result.append([z/self._n_estimators for z  in Counter(prob_list).values()])
        return np.array(result)


