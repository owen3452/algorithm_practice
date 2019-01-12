#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zhen
# @email    : zhendch@qq.com

import numpy as np
import matplotlib.pyplot as plt
import collections

def list_distinct(list):
    #list 去重 降序
    l = collections.Counter(list)
    return sorted(l.keys(), reverse=True)

class DecisionTree(object):
    """
    type: distrete, continuous
    criterion: gini,entropy
    splitter: best, random
    max_depth: 最大深度
    min_impurity_decrease： 熵or基尼系数减少的阈值，小于阈值不分裂
    min_impurity_split： 节点熵or基尼系数的阈值，小于阈值不分裂
    min_samples_split： 节点样本数的阈值，小于阈值不分裂
    """

    def __init__(self, type='distrete', criterion='gini', splitter='best', min_impurity_decrease=0.0,
                 min_impurity_split=0.0, min_samples_split=1.0, max_depth=None, Columns=[]):
        self._type = type
        self._criterion = criterion
        self._splitter = splitter
        self._min_impurity_decrease = min_impurity_decrease
        self._min_impurity_split = min_impurity_split
        self._min_samples_split = min_samples_split
        self._max_depth = max_depth
        self._Columns = Columns

    class TreeNode(object):
        """
        TreeNode是DecisionTree类内的类 用于存储数据 生命周期为一个DecisionTree实例内 变量暴露给DecisionTree
        has_calc_col 已处理列
        last_impurity 上个节点的impurity
        best_feature 该节点最好的特征
        impurity 该节点的impurity
        is_leaf 是否是叶子节点 是的话存储分类信息
        cate_dic 存储分裂后的类
        """

        def __init__(self, has_calc_col=[], features=None, labels=None, impurity=None, best_feature=None,
                    next_impurity=None, is_leaf=None, nodes_data=[], next_node=None):
            self.has_calc_col = has_calc_col
            self.impurity = impurity
            self.features = features
            self.labels = labels
            self.best_feature = best_feature
            self.next_impurity = next_impurity
            self.is_leaf = is_leaf
            self.nodes_data = nodes_data
            self.next_node = next_node

    def gini(self, y: list):
        """
        计算数组的gini指数
        """
        data = collections.Counter()
        data.update(y)
        n = len(y)
        gini_value = 0.0
        for i in data.values():
            p = i / n * 1.0
            temp = p - p**2
            gini_value += temp
        return gini_value

    def gini_condition(self, X: list, y: list):
        """
        计算条件的gini指数
        i: 用于还原数据集的index
        """
        future_len = len(X)
        gini_dict = collections.defaultdict(list)
        index_dict = collections.defaultdict(list)
        for i in range(len(y)):
            gini_dict[X[i]].append(y[i])
            index_dict[X[i]].append(i)
        gini_index = 0.0
        for value in gini_dict.values():
            gini_index += len(value) / future_len * self.gini(value)
        return {gini_index: index_dict}

    def entropy(self, y: list) -> float:
        """
        计算数组的熵
        """
        result = collections.Counter()
        result.update(y)
        rows_len = len(y)
        assert rows_len  # 数组长度不能为0
        # 开始计算熵值
        ent = 0.0
        for value in result.values():
            p = float(value) / rows_len
            ent -= p * np.math.log2(p)
        return ent

    def entropy_condition(self, X: list, y: list) -> float:
        """
        计算条件熵
        i: 用于还原数据集的index
        """
        entropy_dict = collections.defaultdict(list)  # {0:[], 1:[]}
        index_dict = collections.defaultdict(list)
        for i in range(len(y)):
            entropy_dict[X[i]].append(y[i])
            index_dict[X[i]].append(i)
        # 计算条件熵
        ent = 0.0
        future_len = len(X)  # 数据个数
        for value in entropy_dict.values():
            p = len(value) / future_len * self.entropy(value)
            ent += p
        return {ent: index_dict}

    def choose_best_feature(self, X, y, has_calc_col):
        """
        寻则最佳分裂的属性
        X: dict {a:[]}  y: list
        """
        # 待处理col
        wait_col = list(X.keys())
        for col in has_calc_col:
            wait_col.remove(col)
            # 选择损失函数
        if self._criterion == 'gini':
            loss = self.gini
            cond_loss = self.gini_condition
        elif self._criterion == 'entropy':
            loss = self.entropy
            cond_loss = self.entropy_condition
        else:
            raise Exception("Wrong Value: criterion")
        if len(wait_col):
            # 离散值
            impurity_dic = collections.defaultdict(dict)
            # {feaure: {impurity: {feaure_vlue1: [data], feaure_vlue2: [data], ...}}}
            if self._type == 'distrete':
                for key in wait_col:
                    col_new = list(map(lambda x: '==' + str(x) , X[key]))
                    impurity_dic[key].update(cond_loss(col_new, y))
            # 连续值
            elif self._type == 'continuous':
                for key in wait_col:
                    l = list_distinct(X[key])
                    temp_impurity = collections.defaultdict(dict)
                    # {impurity: {feaure_vlue: data}} 选最小的
                    if len(l) > 1:
                        for i in range(len(l) - 1):
                            mid_n = (l[i] + l[i + 1])*1.0 / 2
                            col_new = list(map(lambda x: '>=' + str(mid_n) if x > mid_n else '<=' + str(mid_n), X[key]))
                            temp_impurity.update(cond_loss(col_new, y))
                    else:
                        col_new = list(map(lambda x: '==' + str(x), X[key]))
                        temp_impurity.update(cond_loss(X[key],y))
                    temp_best_feature = sorted(temp_impurity.items(), key=lambda x: x[0])[0]
                    impurity_dic[key] = {temp_best_feature[0]: temp_best_feature[1]}
            impurity_dic = sorted(impurity_dic.items(), key=lambda x: x[1].keys())[0]
            # 处理输出值
            impurity = loss(y)
            best_feature = impurity_dic[0]
            next_impurity = impurity_dic[1].keys()
            # 输出分裂后数据集
            feature_index = list(impurity_dic[1].values())[0]
            #impurity_dic[1].values() 后面报错 'dict_values' object has no attribute 'values'
            # y_out: {feature_vlue1: [y]}
            y_out = collections.defaultdict(list)
            for i in feature_index.keys():
                y_out[i] = list(map(lambda n: y[n], feature_index[i]))
            # X_out: {feature_vlue1: feature1:[x1], feature2:[x2]}
            X_out = collections.defaultdict(dict)
            for i in feature_index.keys():
                dic_temp = {}
                for j in X.keys():
                    dic_temp.update({j: list(map(lambda n: X[j][n], feature_index[i]))})
                X_out[i] = dic_temp
            return impurity, best_feature, float(list(next_impurity)[0]), X_out, y_out
        else:
            return None, None, None, None, None

    def is_leaf_node(self, tree_node):
        """
        是否是叶子节点
        """
        result = None
        n = self._min_impurity_decrease
        if len(list_distinct(tree_node.labels)) == 1:
            # 样本已分净
            result = list_distinct(tree_node.labels)[0]
        elif len(tree_node.features.keys())==len(tree_node.has_calc_col) \
             or len(tree_node.has_calc_col)==self._max_depth \
             or tree_node.impurity <= self._min_impurity_split  \
             or tree_node.impurity - tree_node.next_impurity <= self._min_impurity_split \
             or len(tree_node.labels) <= (n if n >= 1 else n * self._all_data_cnt):
            # 达到三种阈值
            result = sorted(collections.Counter(tree_node.labels).items(), key=lambda x: x[1], reverse=True)[0][0]
        return result

    def build_tree(self, tree_node):
        """
        建立递归决策树
        """
#         print(tree_node,'\n',
#               tree_node.has_calc_col,'\n',
#               tree_node.best_feature,'\n',
#               tree_node.is_leaf,'\n')
        # 判断是否跳出递归
        if tree_node.is_leaf is None:
            for i in tree_node.nodes_data[0].keys():
                (impurity, best_feature, next_impurity, features, labels) = \
                self.choose_best_feature(tree_node.nodes_data[0][i], tree_node.nodes_data[1][i], tree_node.has_calc_col+[tree_node.best_feature])
                temp_tree = self.TreeNode(tree_node.has_calc_col+[tree_node.best_feature], \
                                          tree_node.nodes_data[0][i], tree_node.nodes_data[1][i], \
                                          impurity,  best_feature, next_impurity, None, [features, labels], {})
                temp_tree.is_leaf = self.is_leaf_node(temp_tree)
                tree_node.next_node.update({i: temp_tree})
#                 print(tree_node,'\n',
#                       tree_node.next_node,'\n',
#                       '------------------','\n',
#                       temp_tree,'\n',
#                       temp_tree.has_calc_col,'\n',
#                       temp_tree.best_feature,'\n',
#                       temp_tree.is_leaf,'\n')
                self.build_tree(temp_tree)

    def fit(self, X, y):
        m, n = np.shape(X)
        self._Columns = [str(col) for col in range(n)] if self._Columns == [] else self._Columns
        X = {col: list(x) for col, x in zip(self._Columns, X.T)}
        self._all_data_cnt = len(y)
        #self._root_node = self.TreeNode(features=X, labels=y)
        (impurity, best_feature, next_impurity, features, labels) = \
            self.choose_best_feature(X, y, [])
        self._tree_node = self.TreeNode([], X, y, impurity, best_feature, next_impurity,\
                                        None, [features, labels], {})
        self._tree_node.is_leaf = self.is_leaf_node(self._tree_node)
        self.build_tree(self._tree_node)

    def predict_one(self, X, tree_node=None):
        if tree_node==None:
            tree_node = self._tree_node
        if tree_node.is_leaf!=None:
            return tree_node.is_leaf
        else:
            for condition in tree_node.next_node.keys():
                if eval(str(X[tree_node.best_feature][0]) + condition):
#                     print(tree_node.best_feature, str(X[tree_node.best_feature][0]) + condition)
                    return self.predict_one(X, tree_node.next_node[condition])

    def predict(self, X):
        result = []
        for x in X:
            x = {col: [x] for col, x in zip(self._Columns, x.T)}
            result.append(self.predict_one(x))
        return np.array(result)




# """
# #测试
# """
#
# from sklearn.datasets import load_iris
# import pandas as pd
#
# iris = load_iris()
# #print(iris.feature_names)  # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# #print(iris.target_names) #['setosa' 'versicolor' 'virginica']
# #print(iris.data)
# #print(iris.target)
# df_feature = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
# df_label = pd.DataFrame(iris.target, columns=['label'])
# df = pd.concat([df_feature, df_label], axis=1)
# df = df[df['label'] <= 1]
# df
#
# test = DecisionTree(type='continuous')
# #test.entropy_condition([1,1,0,0], [1,0,0,0])
# X = df.iloc[0:98,0:4].to_dict(orient= 'list')
# y = list(df.iloc[0:98,[4]].to_dict(orient= 'list').values())[0]
#
# X_test = df.iloc[[99],0:4].to_dict(orient= 'list')
# y_result = list(df.iloc[[99],[4]].to_dict(orient= 'list').values())[0]
#
# test.fit(X,y)
# #test.predict_one(X_test)
# #1
# test._tree_node
#
# ##画图
# from graphviz import Digraph
#
# dot = Digraph(comment='The Test Table', format="png")
# def scan_class(name, tree):
#     dot.node(name, str(tree.features +
#                     tree.labels +
#                     tree.has_calc_col +
#                     tree.impurity +
#                     tree.best_feature +
#                     #tree.next_impurity +
#                     tree.is_leaf ) )
#     if tree.next_node==None:
#         return
#     for i in tree.next_node.keys():
#         scan_class(i, tree.next_node[i])
#         dot.edge(name, i, constraint='false')
#
# scan_class('init', test._tree_node)
#
# # 保存source到文件，并提供Graphviz引擎
# dot.save('test-table.gv')
# dot.render('test-table.gv')
# dot.view()


      
