#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zhen
# @email    : zhendch@qq.com

import numpy as np
from scipy import stats

class naive_bayes(object):
    '''
    feature_type: distrete, continuous
    distribution: Gaussian Exponential
    '''
    def __init__(self, feature_type='distrete', distribution='Gaussian'):
        self._feature_type = feature_type
        self._distribution = distribution

    def handle_gaussian(self, X, x):
        # 很多博客反馈scipy的fit不准，待改进
        u, sigma = stats.norm.fit(X)
        return stats.norm.pdf(self, x, u, sigma)

    def handle_exponential(self, X, x):
        # 很多博客反馈scipy的fit不准，待改进
        loc, scale = stats.expon.fit(X)
        return stats.expon(self, x, loc, scale)

    def fit(self, X, y):




