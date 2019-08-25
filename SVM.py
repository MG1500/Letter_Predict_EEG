from sklearn import svm
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import random
feat = pd.read_csv("features.csv")
val = pd.read_csv("values.csv")
feat = np.array(feat)
val = np.array(val)
feat = feat[:590]
val = val[:590,0]
train_feat=[]
train_val=[]
test_feat=[]
test_val=[]
split=0.87
for i in range(len(feat)):
    if random.random()<split:
        train_feat.append(feat[i])
        train_val.append(val[i])
    else:
        test_feat.append(feat[i])
        test_val.append(val[i])
clf = svm.SVC(gamma='scale')
clf.fit(train_feat,train_val)
correct=0
for i in range(len(test_feat)):
    x=clf.predict([test_feat[i][0:len(test_feat[i])]])
    if x==test_val[i]:
        correct+=1
print((correct/float(len(test_feat)))*100)

