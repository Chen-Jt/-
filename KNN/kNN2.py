# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 19:25:56 2018

@author: cmoyan
@Last Modified by:   cmoyan
# @Last Modified time: Fri Jul 20 19:25:56 2018
"""

import numpy as np
import os
from kNN0 import *

def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handWritingClassTest():
    hwLabels = []
    dataDirPath = 'trainingDigits'
    trainingFileList = os.listdir(dataDirPath)
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileName = trainingFileList[i]
        classNameStr = int(fileName.split('_')[0]) 
        hwLabels.append(classNameStr)
        trainingMat[i,:] = img2vector(dataDirPath+'/%s' %  fileName)
    testFileList = os.listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileName = testFileList[i]
        classNameStr = int(fileName.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileName)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("The classifier came back with: %d,the real answer is: %d" % (classifierResult,classNameStr))
        if(classifierResult != classNameStr):
            errorCount += 1.0
    print("The total number of errors is: %d" % errorCount)
    print("The total error rate is: %f" % (errorCount/float(mTest)))
            
if __name__ == '__main__':
    handWritingClassTest()