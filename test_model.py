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

directory_positive = './test/exampleImages/'
directory_negative = './test/exampleImages/'
model_pkl_file = "clf_classifier_model.pkl"

with open(model_pkl_file, 'rb') as file:
	clf = pickle.load(file)

# load patches
print('Loading patches...   ')
m = loadmat('./universal_patch_set.mat')
patches = [patch.reshape(4, 4, 4) for patch in m['patches'][0][1].T[0:100]]

count_positive = 0
for filename in os.listdir(directory_positive):
 if filename.endswith(".jpg"):
 	count_positive += 1

# compute positive test responses
print('Testing on {} faces...'.format(count_positive))
results1 = [clf.predict(np.array(computeResponses(mpimg.imread(directory_positive+filename)[:,:,1],patches)).reshape(1,-1))[0] for filename in os.listdir(directory_positive) if filename.endswith(".jpg")]
print('Accuracy: {}/{}'.format(sum(results1), count_positive))

count_negative = 0
for filename in os.listdir(directory_negative):
 if filename.endswith(".jpg"):
 	count_negative += 1

# compute negative test responses
print('Testing on {} faces...'.format(count_negative))
results2 = [clf.predict(np.array(computeResponses(mpimg.imread(directory_negative+filename)[:,:,1],patches)).reshape(1,-1))[0] for filename in os.listdir(directory_negative) if filename.endswith(".jpg")]
print('Accuracy: {}/{}'.format(count_negative-sum(results2), count_negative))