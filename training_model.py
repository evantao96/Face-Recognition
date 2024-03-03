import math
import numpy as np
from scipy import signal, ndimage
from scipy.io import loadmat
from sklearn import svm
import pickle
import matplotlib.image as mpimg
from sklearn.preprocessing import normalize
import os
from classify import *

directory_positive = './training/train_positive_2/'
directory_negative = './training/train_negative_2/'
model_pkl_file = "clf_classifier_model.pkl"

# load patches
print('Loading patches...   ')
m = loadmat('./universal_patch_set.mat')
patches = [patch.reshape(4, 4, 4) for patch in m['patches'][0][1].T[0:100]]

count_positive = 0
for filename in os.listdir(directory_positive):
 if filename.endswith(".jpg"):
    count_positive += 1

# compute positive test responses
print('Training on {} faces...'.format(count_positive))
X1 = [computeResponses(mpimg.imread(directory_positive+filename)[:,:,1],patches) for filename in os.listdir(directory_positive) if filename.endswith(".jpg")]

count_negative = 0
for filename in os.listdir(directory_negative):
 if filename.endswith(".jpg"):
    count_negative += 1

# compute negative training responses
print('Training on {} non-faces...'.format(count_negative))
X2 = [computeResponses(mpimg.imread(directory_negative+filename)[:,:,1], patches) for filename in os.listdir(directory_negative) if filename.endswith(".jpg")]

# classify
X = np.concatenate((X1, X2))
y = np.array([1]*count_positive + [0]*count_negative)
clf = svm.SVC()
clf.fit(X, y)

with open(model_pkl_file, 'wb') as file:
    pickle.dump(clf, file)