# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 11:19:20 2018

@author: cmoyan
"""
import numpy as np
import matplotlib.pyplot as plt
from kNN0 import *
#从文件读数据进矩阵
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines,3))
    classLabelVeactor = []
    index = 0
    for line in arrayOLines:
        listFormLine = line.strip().split('\t')
        returnMat[index,:] = listFormLine[0:3]
        classLabelVeactor.append(int(listFormLine[-1]))
        index+=1
    return returnMat,classLabelVeactor

#画图
def drawplot(dataX,dataY,colorX,colorY):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataX,dataY,colorX,colorY)
    plt.show()

#归一化特征值
def autoNorm(dataSet):
    minVal = dataSet.min(0) #0  按列比较 1 按行比较
    maxVal = dataSet.max(0)
    ranges = maxVal - minVal
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVal,(m,1))
    normDataSet /= np.tile(ranges,(m,1))
    return normDataSet, ranges, minVal

def dataClassTest():
    hoRatio = 0.10
    dataMat,dataLabels = file2matrix("datingTestSet2.txt")
    normMat,ranges,minVal  = autoNorm(dataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio) #测试集数目
    errorCount = 0.0
    for i in range(numTestVecs):
       classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],dataLabels[numTestVecs:m],4)
       print("【%d】\tThe classifier came back with:%d,the real answer is %d" % (i+1,classifierResult,dataLabels[i]))
       if classifierResult != dataLabels[i]:
           errorCount += 1
    print("The total error rate is %f. " % (errorCount/float(numTestVecs)))   

def classifyPerson():
    resultList = ['not at all','in some doses','in large doese']
    ffMiles = float(input("Frequent flier miles earned per years?\n"))
    percentTats = float(input("Percentage of time spent playing video games?\n"))
    iceCream = float(input("Liters of ice cream consumed per years?\n"))
    dataMat,dataLabels = file2matrix("datingTestSet2.txt")
    normMat,ranges,minVal  = autoNorm(dataMat)
    inArr = (np.array([ffMiles,percentTats,iceCream]) - minVal ) / ranges
    classifierResult = classify0(inArr,normMat,dataLabels,41)
    
    print("You will probably like this person:",resultList[classifierResult - 1])
    
       
if __name__ == '__main__':
#    dataMat,dataLabels = file2matrix("datingTestSet2.txt")
#    print(dataMat[:,1:5])
#    print(dataLabels[0:20])
#    drawplot(dataMat[:,1],dataMat[:,2],15.0*np.array(dataLabels),15.0*np.array(dataLabels))
#    drawplot(dataMat[:,0],dataMat[:,1],15.0*np.array(dataLabels),15.0*np.array(dataLabels))
#    normData  = autoNorm(dataMat)
    #print(normData)
#    dataClassTest()
    classifyPerson()