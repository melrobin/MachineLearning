import numpy as np
from sklearn import metrics, svm

raw_data = np.genfromtxt('plrx.txt')

data = raw_data[:,:12]
label = raw_data[:,-1]



