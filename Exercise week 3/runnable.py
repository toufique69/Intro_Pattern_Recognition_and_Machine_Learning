import pickle
import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
from random import random
import math

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

def class_acc(pred,gt):
    errorSum=0;
    accuracyOfModal=0;
    for i in range(0,len(gt)):
        if gt[i]!=pred[i]:
            errorSum=errorSum+1;
    accuracyOfModal=(pred.shape[0]-errorSum)/(pred.shape[0])
    accuracyPercentage=accuracyOfModal*100
    return accuracyPercentage

def cifar10_classifier_random(x):
    y=randint(0, 9, x.shape[0])
    
    return y

def cifar10_classier_1nn(X,trdata,trlabels,predY):
    print(trdata.shape)
    print(X.shape)
    predLabel=predY[0:100];
    minDistance=np.full(100, 1000000000)
    imageCount=0;
    labelIndex=0;
    comparedImageCount=0;
    euclidentDistance=0;
    for i in X[0:100,:]:
        euclidentDistance=0;
        comparedImageCount=0;
        for j in trdata[0:100,:]:
            euclidentDistance=abs(sum(sum((X[i]-trdata[j]))))
            if euclidentDistance < minDistance[imageCount]:
                minDistance[imageCount]=euclidentDistance
                print(str(minDistance[imageCount])+" minDistance")
                labelIndex=trlabels[comparedImageCount];
                print(str(labelIndex)+" labelIndex")
                print("-----------------------------------------------------------------")
                comparedImageCount=comparedImageCount+1;
                if minDistance[imageCount] == 0.0:
                  break
        predLabel[imageCount]=labelIndex
        print(predLabel)
        print("/********************************** new image "+str(imageCount)+" *******************************************/")
        
        imageCount=imageCount+1;
    
    print(predLabel)
    return predLabel,minDistance

datadict = unpickle('D:\TAU\Courses\Intro to ML\Exercise week 3\cifar-10-batches-py/test_batch')

X = datadict["data"]
Y = datadict["labels"]

labeldict = unpickle('D:\TAU\Courses\Intro to ML\Exercise week 3\cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]

Y = np.array(Y)

predY=np.full(X.shape[0], 0)
minDistance=np.full(10, 0)
#batchArr=[2,3,4,5]
batchArr=[3]
for i in batchArr:
    datadict = unpickle('D:\TAU\Courses\Intro to ML\Exercise week 3\cifar-10-batches-py/data_batch_'+str(i))
    trdata = datadict["data"]
    trlabels = datadict["labels"]
    predictedY,minDistance=cifar10_classier_1nn(X,trdata,trlabels,predY)

predY=cifar10_classifier_random(X);
randomAccuracy=class_acc(predY,Y)
Y=Y[0:100];
print("Predicted Y set is Given below")
print(predY)
print("Actual Y set is Given below")
print(Y)
predictedY=predictedY[0:100]
accuracyPercentage=class_acc(predictedY,Y)
print(minDistance)
print("Random accuracy: "+str(randomAccuracy))
print("Total Accuracy: "+str(accuracyPercentage))