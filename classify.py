import numpy as np
from sklearn import neighbors, metrics
from sklearn import svm, metrics

raw_data = np.genfromtxt('plrx.txt')		#reading data from txt file

X= raw_data[:,:12]				#splicing the raw data
y = raw_data[:,-1]
#print data.shape, label.shape			#shapes of the matrices

clf1 = neighbors.KNeighborsClassifier(n_neighbors= 2)	#nearest two neighbors
#clf2 = svm.SVC(kernel = 'rbf' )		
clf2 = svm.SVC(gamma=5 )		
#defining the classifier. Kernel= 'rbf', gamma= 0.083
clf1.fit(X, y)	#fitting the data with the model
clf2.fit(X, y)	#fitting data with model
result_knn = clf1.predict(X)	#prediction with training data
result_svm = clf2.predict(X)	#prediction of training data

training_accuracy_knn = metrics.accuracy_score(y, result_knn)	
training_accuracy_svm = metrics.accuracy_score(y, result_svm)	
#training accuracy for nearest neighbors
c_matrix_knn = metrics.confusion_matrix(y, result_knn)		
c_matrix_svm = metrics.confusion_matrix(y, result_svm)		
#confusion matrix
c_matrix_svm = metrics.confusion_matrix(y, result_svm)	#confusion matrix
print "kNN training accuracy:",training_accuracy_knn
print "SVM training accuracy:",training_accuracy_svm
print "Confusion matrix for knn:\n", c_matrix_knn
print "Confusion matrix for svm:\n", c_matrix_svm
