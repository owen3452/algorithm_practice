import numpy as np
import math
import matplotlib.pyplot as plt
%matplotlib inline

#LR概率函数 w:(w,b), features:(features vector,1)
def sigmoid_func(features, w):
    z = np.dot(w.T, features)*(-1)
    value = 1/(1 + np.e**(np.dot(w.T, features)*(-1)))
    return value
    
#损失函数
def lost_func(samples, w):
    lost = 0
    for i in samples:
        temp = np.dot(w.T, np.append(i[:-1], 1))*i[-1]*(-1) + np.log(np.e**(np.dot(w.T, np.append(i[:-1],1)))+1)
        lost = lost + temp
    return lost

#梯度向量
def gradient(samples, w):
    gradient_value = np.zeros(len(w), dtype='float')
    for i in samples:
        temp = np.append(i[:-1], 1)*(sigmoid_func(np.append(i[:-1], 1),w) - i[-1])
        gradient_value = gradient_value + temp
    return gradient_value
    
#构造数据samples 0：均值[0,0] 二维正态分布数据 1：均值[1,4] 二维正态分布数据
np.random.seed(12)
num_observations = 5000

x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
simulated_labels = np.hstack((np.zeros(num_observations),
                              np.ones(num_observations)))
plt.figure(figsize=(12,8))
plt.scatter(simulated_separableish_features[:, 0], simulated_separableish_features[:, 1],
            c = simulated_labels, alpha = .4)
samples = np.vstack((np.hstack((x1, np.array([[0]]*len(x1)))), np.hstack((x2, np.array([[1]]*len(x2))))))

# 初始化w [-7 ,10,-15] , 步长alpha = -0.01
w = np.array([-7 ,10,-15])
lost_value = lost_func(samples, w)
print (w,
       lost_value)
alpha = -0.01
# 循环 2000次
for l in range(2000):
    gradient_eg = gradient(samples, w)
    w = w + gradient_eg*alpha
    lost_value = lost_func(samples, w)
    print (w, '/n', lost_value)
'''
lost_value 最小
[ -5.61764839   9.05269005 -15.62154972] /n 143.00137622885316
'''

# 验证得到的最优的参数
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(fit_intercept=True, C = 1e15)
clf.fit(samples[:,:-1], samples[:,-1])
print(clf.coef_,clf.intercept_,)
'''
[[-5.02712589  8.23286817]] [-13.99400825]
'''
