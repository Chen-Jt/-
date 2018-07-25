# -*- coding: utf-8 -*-
# @Author: cjt
# @Date:   2018-07-19 20:13:02
# @Last Modified by:   cjt
# @Last Modified time: 2018-07-20 10:17:31
from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels
'''
inX     输入向量
dataSet 训练集特征向量 
labels  训练集标签列表  与训练集特征向量顺序相对应
k       int型，即为K近邻中的K值
'''
def classify0(inX, dataSet, labels, k):
    # 计算输入值与训练集的距离
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)  # axis=1 按行相加
    distances = sqDistances ** 0.5  # sqrt(0.5)

    # 选择距离最小的K个点
    sortedDistIndicies = distances.argsort()  # 按值升序排序 返回数组索引
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 选取出现次签最多的标签
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

if __name__ == '__main__':
    group,labels = createDataSet()
    res = classify0([0,0],group,labels,3)
    print(res)



