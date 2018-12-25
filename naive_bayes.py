#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zhen
# @email    : zhendch@qq.com

import numpy as np
from scipy import stats,optimize

class naive_bayes(object):
    '''
    feature_type: distrete, continuous
    distribution: Gaussian Exponential
    '''
    def __init__(self, feature_type='distrete', distribution='Gaussian'):
        self._feature_type = feature_type
        self._distribution = distribution

    def gaussian(self, x, sigma, u):
        y = np.exp(-(x - u) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
        return y

    def exponential(self, x, a, b, c):
        return a * np.exp(-b * x) + c

    def handle_gaussian(self, X, x):
        u, sigma = stats.norm.fit(X)
        return self.gaussian(self, x, sigma, u)

    def handle_exponential(self, X, x):
        popt, pcov = optimize.curve_fit(self.exponential, xdata, ydata)
        return self.gaussian(self, x, sigma, u)



