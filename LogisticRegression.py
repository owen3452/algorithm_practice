#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zhen
# @email    : zhendch@qq.com

import numpy as np
import random
import matplotlib.pyplot as plt


class LogisticRegression(object):
    '''
    penalty： 正则化 L1/L2
    C: 正则化系数
    tol: 停止迭代 最小值
    max_iter： 最大迭代次数
    learning_rate：步长
    solver: sgd/bgd
    '''
    def __init__(self, penalty='l2', C=1.0, tol=0.0001, max_iter=100, learning_rate=0.01, solver='sgd'):
        self._penalty = penalty
        self._tol = tol
        self._C = C
        self._max_iter = max_iter
        self._solver = solver
        self._learning_rate = learning_rate

    def sigmoid_func(self, X, w):
        # sigmoid函数 w:(w,b), X:(features vector,1)
        z = np.dot(w.T, X)*(-1)
        value = 1.0/(1 + np.e**(z))
        return value

    def lost_func(self, X, y, w):
        # 损失函数
        lost = 0
        n = len(X)
        # 惩罚项
        penalty = sum(map(abs, w)) if self._penalty=='L1' else sum(map(lambda x: 0.5*x**2, w))
        for i in zip(X, y):
            temp = np.dot(w.T, np.append(i[0], 1))*i[1]*(-1.0) + np.log(np.e**(np.dot(w.T, np.append(i[0],1)))+1)
            lost = lost + temp
        return (lost*1.0+penalty*self._C)/n

    def gradient_bgd(self, X, y, w):
        # 梯度向量
        n = len(X)
        # 惩罚项梯度
        penalty_victor = map(lambda x: self._C if x>0 else self._C*(-1), w) if self._penalty == 'L1' \
                         else map(lambda x: self._C * x if x > 0 else self._C * (-1) * x, w)
        gradient_victor = np.zeros(len(w), dtype='float')
        for i in zip(X, y):
            temp = np.append(i[0], 1)*(i[1] - self.sigmoid_func(np.append(i[0], 1),w))
            gradient_victor = gradient_victor + temp
        div = np.full_like(gradient_victor, n, dtype=np.double)
        return (gradient_victor-penalty_victor) / div

    def gradient_sgd(self, X, y, w):
        # 梯度向量 X y为一组feature和label
        n = len(X)
        # 惩罚项梯度
        penalty_victor = map(lambda x: self._C if x > 0 else self._C * (-1), w) if self._penalty == 'L1' \
            else map(lambda x: self._C * x if x > 0 else self._C * (-1) * x, w)
        gradient_victor = np.zeros(len(w), dtype='float')
        temp = np.append(X, 1) * (y - self.sigmoid_func(np.append(X, 1), w))
        gradient_victor = gradient_victor + temp
        div = np.full_like(gradient_victor, n, dtype=np.double)
        return (gradient_victor - penalty_victor) / div

    def fit(self, X, y):
        m, n = np.shape(X)
        w = np.random.randn(n)
        lost = []
        lost.append(self.lost_func(X, y, w))
        for i in range(self._max_iter):
            if self._solver=='bgd':
                gradient_eg = self.gradient_bgd(X, y, w)
                w += gradient_eg * self._learning_rate
                lost.append(self.lost_func(X, y, w))
            elif self._solver=='sgd':
                rand = random.randint(0, m-1)
                gradient_eg = self.gradient_bgd(X[rand], y[rand], w)
                w += gradient_eg * self._learning_rate
                lost.append(self.lost_func(X, y, w))
            if lost[i+1] - lost[i] <= self._tol:
                break
        self._w = w

    def predict_prob(self, X):
        return map(lambda x: self.sigmoid_func(x, self._w), X)


# #构造数据samples 0：均值[0,0] 二维正态分布数据 1：均值[1,4] 二维正态分布数据
# np.random.seed(12)
# num_observations = 5000
#
# x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
# x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)
#
# simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
# simulated_labels = np.hstack((np.zeros(num_observations),
#                               np.ones(num_observations)))
# plt.figure(figsize=(12,8))
# plt.scatter(simulated_separableish_features[:, 0], simulated_separableish_features[:, 1],
#             c = simulated_labels, alpha = .4)
# samples = np.vstack((np.hstack((x1, np.array([[0]]*len(x1)))), np.hstack((x2, np.array([[1]]*len(x2))))))
#
# # 初始化w [-7 ,10,-15] , 步长alpha = -0.01
# w = np.array([-7 ,10,-15])
# lost_value = lost_func(samples, w)
# print (w,
#        lost_value)
# alpha = -0.01
# # 循环 2000次
# for l in range(2000):
#     gradient_eg = gradient(samples, w)
#     w = w + gradient_eg*alpha
#     lost_value = lost_func(samples, w)
#     print (w, '/n', lost_value)
# '''
# lost_value 最小
# [ -5.61764839   9.05269005 -15.62154972] /n 143.00137622885316
# '''
#
# # 验证得到的最优的参数
# from sklearn.linear_model import LogisticRegression
#
# clf = LogisticRegression(fit_intercept=True, C = 1e15)
# clf.fit(samples[:,:-1], samples[:,-1])
# print(clf.coef_,clf.intercept_,)
# '''
# [[-5.02712589  8.23286817]] [-13.99400825]
# '''
