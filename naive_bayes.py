#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zhen
# @email    : zhendch@qq.com

import numpy as np
import collections
from scipy import stats

class naive_bayes(object):
    '''
    feature_type: distrete, continuous
    distribution: Gaussian, Exponential
    '''
    def __init__(self, feature_type='distrete', distribution='Gaussian'):
        self._feature_type = feature_type
        self._distribution = distribution
        self._condition_prob = collections.defaultdict(dict)
        # condition_prob
        # conti: {c1:{c1:p,x11:[u,var],x21:[u,var],...},
        #         c2:{c2:p,x12:[u,var],x22:[u,var],...},
        #         ...}
        # distr: {c1:{c1:p,x11:{a:p,b:p},x21:{a:p,b:p},...},
        #         c2:{c2:p,x12:{a:p,b:p},x22:{a:p,b:p},...},
        #         ...}

    def handle_gaussian(self, X):
        loc, scale = stats.norm.fit(X)
        # loc: avg scale: var
        return [loc, scale]

    def handle_exponential(self, X):
        # 很多博客反馈stats.expon使用最大似然拟合的效果不好
        loc, scale = stats.expon.fit(X)
        # loc: unknown scale: lambda
        return [loc, scale]

    def gaussian(self, x, loc, scale):
        return np.exp(stats.norm.pdf(x, loc, scale)) # prob取对数 防止内存溢出

    def exponential(self, x, loc, scale):
        return np.exp(stats.norm.pdf(x, loc, scale)) # prob取对数 防止内存溢出

    def handle_distrete_prob(self, X):
        # 拉普拉斯平滑后的离散值概率
        X = np.array(X)
        prob_dic = {}
        for i in np.unique(X):
            X_tmp = X[np.where(X == i)]
            prob_dic[i] = (len(X_tmp) + 1) * 1.0/(len(X) + len(np.unique(X)))
        prob_dic['unknown'] = 1.0/(len(X) + len(np.unique(X))) # 训练集中未出现的特征
        return prob_dic

    def distrete_prob(self, x, d):
        # d: dict
        try:
            P = d[x]
        except:
            P = d['unknown']
        return np.exp(P) # prob取对数 防止内存溢出

    def fit(self, X, y):
        # P(c),P(xj|ci)存储在self._condition_prob
        X = np.array(X)
        y = np.array(y)
        for i in np.unique(y):
            X_tmp = X[np.where(y == i)]
            y_tmp = y[np.where(y == i)]
            self._condition_prob[i].update({i: (len(X_tmp) + 1) * 1.0/(len(y_tmp) + len(np.unique(y)))})
            if self._feature_type == 'continuous':
                m, n = np.shape(X_tmp)
                for j in range(n):
                    Xj_ci = [x[j] for x in X_tmp]
                    if self._distribution == 'Gaussian':
                        self._condition_prob[i].update({'x' + str(j) + str(i): self.handle_gaussian(Xj_ci)})
                    elif self._distribution == 'Exponential':
                        self._condition_prob[i].update({'x' + str(j) + str(i): self.handle_gaussian(Xj_ci)})
                    else:
                        raise Exception("ValueError: Unexpected Parameter 'distribution")
            elif self._feature_type == 'distrete':
                m, n = np.shape(X_tmp)
                for j in range(n):
                    Xj_ci = [x[j] for x in X_tmp]
                    self._condition_prob[i].update({'x' + str(j) + str(i): self.handle_gaussian(Xj_ci)})

    def predict_prob(self, X):
        X = np.array(X)
        m, n = np.shape(X)
        prob = []
        if self._feature_type == 'continuous':
            for x in X:
                Pc = [p.values()[0] for p in self._condition_prob.values()]
                for j in range(n):
                    if self._distribution == 'Gaussian':
                        Pxj = [self.gaussian(x[j], p.values()[j+1][0], p.values()[j+1][1])
                               for p in self._condition_prob.values()]
                    elif self._distribution == 'Exponential':
                        Pxj = [self.exponential(x[j], p.values()[j + 1][0], p.values()[j + 1][1])
                               for p in self._condition_prob.values()]
                    Pc = np.dot(Pc, Pxj)
                prob_dict = {i[0]: i[1] for i in zip(self._condition_prob.keys(), Pc)}
                prob.append(prob_dict)
        elif self._feature_type == 'distrete':
            for x in X:
                Pc = [p.values()[0] for p in self._condition_prob.values()]
                for j in range(n):
                    Pxj = [self.distrete_prob(x[j], p.values()[j+1]) for p in self._condition_prob.values()]
                    Pc = np.dot(Pc, Pxj)
                prob_dict = {i[0]: i[1] for i in zip(self._condition_prob.keys(), Pc)}
                prob.append(prob_dict)
        return prob

# 测试
# num_observations = 5000
# x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
# x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)
#
# simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
# simulated_labels = np.hstack((np.zeros(num_observations),
#                               np.ones(num_observations)))
#
# model = naive_bayes(feature_type='continuous', distribution='Gaussian')
# model.fit(simulated_separableish_features[:8000], simulated_labels[:8000])
#
# predict = model.predict_prob(simulated_separableish_features[8000:])
#
# predict_y = [1 if y.values()[0]<y.values()[1] else 0 for y in predict]
# y = simulated_labels[8000:]
# sum([1 if a[0]==a[1] else 0 for a in zip(predict_y, y)])*1.0/2000
# 0.9775







