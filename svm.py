#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zhen
# @email    : zhendch@qq.com

import numpy as np


class SVC(object):
    '''
    C: 正则化系数
    kernel: 核函数（linear：线性核, rbf:高斯核) [Kernel, param]
    tol: 停止迭代的最小阈值
    max_iter: max_iter
    '''

    def __init__(self, C=1.0, kernel=['rbf', 1], tol=0.001, max_iter=500):
        self._C = C
        self._kernel = kernel
        self._tol = tol
        self._max_iter = max_iter

    def kernel(self, x1, x2):
        # 核函数 K(x1,x2)
        x1 = np.mat(x1)
        x2 = np.mat(x2)
        if self._kernel[0] == 'linear':  # 线性函数
            K = np.sum(x1 * x2.T)
        elif self._kernel[0] == 'rbf':  # 高斯径向核
            K = np.exp(np.sum((x1-x2) * (x1-x2).T, axis=1, keepdims=True) / (-1 * self._kernel[1] ** 2))
        else:
            raise Exception("ValueError: Unexpected Parameter 'kernel")
        return K

    def kernel_mat(self, X):
        m = np.shape(X)[0]
        kernel_matrix = np.mat(np.zeros((m, m)))
        for i in range(m):
            for j in range(m):
                kernel_matrix[j, i] = self.kernel(X[j], X[i])
        return kernel_matrix

    def cal_error(self, alpha_index_k):
        self.b = 0
        # 误差值的计算
        predict_k = float(np.multiply(self.alphas, self.y).T * self.kernel_mat(self.X)[:, alpha_index_k] + self.b)
        error_k = predict_k - float(self.y[alpha_index_k])
        return error_k

    def select_second_sample_j(self, alpha_index_i, error_i):
        """选择第二个变量
        :param alpha_index_i(float): 第一个变量alpha_i的index_i
        :param error_i(float): E_i
        :return:第二个变量alpha_j的index_j和误差值E_j
        """
        self.error_tmp[alpha_index_i] = [1, error_i]  # 用来标记已被优化
        candidate_alpha_list = np.nonzero(self.error_tmp[:, 0].A)[0]  # 因为是列向量，列数[1]都为0，只需记录行数[0]
        max_step, max_step, error_j = 0, 0, 0

        if len(candidate_alpha_list) > 1:
            for alpha_index_k in candidate_alpha_list:
                if alpha_index_k == alpha_index_i:
                    continue
                error_k = self.cal_error(alpha_index_k)
                if abs(error_k - error_i) > max_step:
                    max_step = abs(error_k - error_i)
                    alpha_index_j, error_j = alpha_index_k, error_k
        else:  # 随机选择
            alpha_index_j = alpha_index_i
            while alpha_index_j == alpha_index_i:
                alpha_index_j = np.random.randint(0, self.n_samples)
            error_j = self.cal_error(alpha_index_j)
        return alpha_index_j, error_j

    def update_error_tmp(self, alpha_index_k):
        '''
        重新计算误差值，并对其标记为已被优化
        :param alpha_index_k: 要计算的变量α
        :return: index为k的alpha新的误差
        '''
        error = self.cal_error(alpha_index_k)
        self.error_tmp[alpha_index_k] = [1, error]

    def choose_and_update(self, alpha_index_i):
        # 判断和选择两个alpha进行更新
        error_i = self.cal_error(alpha_index_i)  # 计算第一个样本的E_i
        if (self.y[alpha_index_i] * error_i < -self._tol) and (self.alphas[alpha_index_i] < self._C) \
                or (self.y[alpha_index_i] * error_i > self._tol) and (self.alphas[alpha_index_i] > 0):
            # 1.选择第二个变量
            alpha_index_j, error_j = self.select_second_sample_j(alpha_index_i, error_i)
            alpha_i_old = self.alphas[alpha_index_i].copy()
            alpha_j_old = self.alphas[alpha_index_j].copy()
            # 2.计算上下界
            if self.y[alpha_index_i] != self.y[alpha_index_j]:
                L = max(0, self.alphas[alpha_index_j] - self.alphas[alpha_index_i])
                H = min(self._C, self._C + self.alphas[alpha_index_j] - self.alphas[alpha_index_i])
            else:
                L = max(0, self.alphas[alpha_index_j] + self.alphas[alpha_index_i] - self.C)
                H = min(self._C, self.alphas[alpha_index_j] + self.alphas[alpha_index_i])
            if L == H:
                return 0
            # 3.计算eta
            eta = self.kernel_mat[alpha_index_i, alpha_index_i] + self.kernel_mat[
                alpha_index_j, alpha_index_j] - 2.0 * self.kernel_mat[alpha_index_i, alpha_index_j]
            if eta <= 0:  # 因为这个eta>=0
                return 0
            # 4.更新alpha_j
            self.alphas[alpha_index_j] += self.y[alpha_index_j] * (error_i - error_j) / eta
            # 5.根据范围确实最终的j
            if self.alphas[alpha_index_j] > H:
                self.alphas[alpha_index_j] = H
            if self.alphas[alpha_index_j] < L:
                self.alphas[alpha_index_j] = L
            # 6.判断是否结束
            if abs(alpha_j_old - self.alphas[alpha_index_j]) < 0.00001:
                self.update_error_tmp(alpha_index_j)
                return 0
            # 7.更新alpha_i
            self.alphas[alpha_index_i] += self.train_y[alpha_index_i] * self.train_y[alpha_index_j] * (
                    alpha_j_old - self.alphas[alpha_index_j])
            # 8.更新b
            b1 = self.b - error_i - self.train_y[alpha_index_i] * self.kernel_mat[alpha_index_i, alpha_index_i] * (
                    self.alphas[alpha_index_i] - alpha_i_old) \
                 - self.y[alpha_index_j] * self.kernel_mat[alpha_index_i, alpha_index_j] * (
                         self.alphas[alpha_index_j] - alpha_j_old)
            b2 = self.b - error_j - self.train_y[alpha_index_i] * self.kernel_mat[alpha_index_i, alpha_index_j] * (
                    self.alphas[alpha_index_i] - alpha_i_old) \
                 - self.y[alpha_index_j] * self.kernel_mat[alpha_index_j, alpha_index_j] * (
                         self.alphas[alpha_index_j] - alpha_j_old)
            if 0 < self.alphas[alpha_index_i] and self.alphas[alpha_index_i] < self.C:
                self.b = b1
            elif 0 < self.alphas[alpha_index_j] and self.alphas[alpha_index_j] < self.C:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0
            # 9.更新error
            self.update_error_tmp(alpha_index_j)
            self.update_error_tmp(alpha_index_i)
            return 1
        else:
            return 0

    def fit(self, X, y):
        #  还是有问题 待修改
        self.n_samples, n = np.shape(X)
        self.X = np.mat(X)
        self.y = np.mat(y)
        self.alphas = np.mat(np.zeros(self.n_samples)) # 初始化alpha [0,0,....,0]
        self.error_tmp = np.mat(np.zeros((self.n_samples, 2)))
        #开始训练
        entireSet = True
        alpha_pairs_changed = 0
        iteration = 0
        while iteration < self._max_iter and (alpha_pairs_changed > 0 or entireSet):
            print("\t iteration: ", iteration)
            alpha_pairs_changed = 0
            if entireSet:  # 对所有样本
                for x in range(n):
                    alpha_pairs_changed += self.choose_and_update(x)
                iteration += 1
            else:  # 对非边界样本
                bound_samples = []
                for i in range(m):
                    if self.alphas[i, 0] > 0 and self.alphas[i, 0] < self.C:
                        bound_samples.append(i)
                for x in bound_samples:
                    alpha_pairs_changed += self.choose_and_update(x)
                iteration += 1
            if entireSet:
                entireSet = False
            elif alpha_pairs_changed == 0:
                entireSet = True
        return self




#####################################################

