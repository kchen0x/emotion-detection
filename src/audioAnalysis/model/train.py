#!/usr/bin/env python
# encoding: utf-8

from sklearn.externals import joblib
from sklearn import svm
import numpy as np
import sys
import os

# 34 stand for short-term features like Zero Crossing Rate,Energy and etc.
stFeatures = 34
# two statistics of the short-term features
statistics = 2
# size of middle-term(group of short-term) 
mSize = 5

# bulid label by filename 
def extractLabel(filename):
    if "an" in filename:
        return 0
    elif "di" in filename:
        return 1
    elif "fe" in filename:
        return 2
    elif "ha" in filename:
        return 3
    elif "sa" in filename:
        return 4
    elif "su" in filename:
        return 5
    elif "ne" in filename:
        return 6

def readNumpyFile(path):
    label = extractLabel(path)
    data = np.load(path)
    return data,label

# read all numpy files from root directory
def getTrainingData(rootDir):
    if not os.path.isdir(rootDir):
        print("Error:Inupt path is't a folder")
        return
    files= os.listdir(rootDir)
    datas = [[]]
    labels = []
    flag = False
    count = 0
    for file in files:
        if ".npy" in file and "st.npy" not in file:
            data,label = readNumpyFile(rootDir+"/"+file)
            try:
                data = data.flatten().reshape((1,stFeatures*mSize*statistics))  
                if flag:
                    datas = np.append(datas,data,axis=0)
                    labels.append(label)
                else:
                    datas = data
                    flag = True
                    labels.append(label)
            except:
                count+=1
                print(file)
    print("data size:%d\tlabel size:%d\tskip:%d"%(len(datas),len(labels),count))
    return datas,labels

if __name__=='__main__':
    rootDir = sys.argv[1]
    print("data extract begin...")
    datas,labels= getTrainingData(rootDir)
    print("data extract end...")
    print("trains begin...")
    clf = svm.SVC(kernel='rbf',max_iter=1000,cache_size=1000)
    clf.fit(datas,labels)
    print("trains end...")
    joblib.dump(clf,rootDir+"/trained_model.m")


