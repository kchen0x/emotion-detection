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

def getTrueValue(filename):
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
            
def predict(rootDir):
    if not os.path.isdir(rootDir):
        print("Error:Input path isn't a folder")
    files = os.listdir(rootDir)
    allNum = len(files)
    count = 0
    clf = joblib.load("trained_model.m")
    for file in files:
        if ".npy" in file:
            trueValue =getTrueValue(file)
            data = np.load(file)
            label = clf.predict(data.flatten().reshape(1,stFeatures*mSize*statistics))
            if label ==trueValue:
                count+=1
    print("total size%d\tprecision%f"%(allNum,float(count)/allNum))

if __name__=='__main__':
    predict(sys.argv[1])
