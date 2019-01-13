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

    def handle_weight(self, X, y, error_list):
        '''
        由于DecisionTree未加入weight，所以adaboosting中将预测错样本weight*e^2（约7）更新权重
        '''
        for x_i, i in zip(X, error_list):
            if i==1:
                np.row_stack((X, x_i, x_i, x_i, x_i, x_i, x_i, x_i))
        for y_i, i in zip(y, error_list):
            if i==1:
                np.append(y, [y_i, y_i, y_i, y_i, y_i, y_i, y_i])
        return X, y

    def fit(self, X, y):
        self.model_list = np.array([])
        self.alpha_list = np.array([])
        for i in range(self._n_estimators):
            tree_tmp = DecisionTree(type=self._type, criterion=self._criterion, splitter=self._splitter,
                                    min_impurity_decrease=self._min_impurity_decrease,
                                    min_impurity_split=self._min_impurity_split,
                                    min_samples_split=self._min_samples_split, max_depth=self._max_depth)
            tree_tmp.fit(X, y)
            predict_tmp = tree_tmp.predict(X)
            error_list = [1 if a != b else 0 for a, b in zip(y, predict_tmp)]
            error_rate = sum(error_list) * 1.0 / len(error_list)
            alpha = 0.5 * np.log(1 / error_rate - 1)
            self.model_list.append(tree_tmp)
            self.alpha_list.append(alpha)
            X, y = self.handle_weight(self, X, y, error_list)

    def predict_prob(self, X):
        result = []
        for x in X:
            predict_list = np.array([model.predict(x) for model in self.model_list])
            result.append(np.dot(predict_list, self.alpha_list.T))
        return np.array(result)


