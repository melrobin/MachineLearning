import numpy as np
from sklearn import ensemble, neighbors, svm, metrics, model_selection

raw_data = np.genfromtxt('plrx.txt')			#reading data from txt file

X= raw_data[:,:12]					#splicing the raw data
y = raw_data[:,-1]

clf1 = neighbors.KNeighborsClassifier(n_neighbors= 2)	#k-NN classifier
clf2 = svm.SVC(gamma = 5)				#SVM classifier
clf3 = ensemble.RandomForestClassifier(n_estimators=10)	#Random Forest classifier

#empty arrays to hold classifier accuracy scores
knn_scores = []
svm_scores= []
trees_scores = []

#Spliting the data into training and testing data
kf = model_selection.KFold(n_splits=5 )
for train , test in kf.split(X):
	clf1.fit(X[train],y[train])
	clf2.fit(X[train],y[train])
	clf3.fit(X[train],y[train])
	acc_knn= clf1.score(X[test], y[test])
	acc_svm= clf2.score(X[test],y[test])
	acc_trees= clf3.score(X[test],y[test])
	knn_scores.append(acc_knn)
	svm_scores.append(acc_svm)
	trees_scores.append(acc_trees)

print "k-NN score: ",100*np.mean(knn_scores)
print"SVM score: ", 100*np.mean(svm_scores)
print "Random Forest score: ", 100*np.mean(trees_scores)

################ Ignore everything below this comment, work from previous script ################
#fitting the data with the model
result_knn = clf1.predict(X)	#prediction with training data
result_svm = clf2.predict(X)	#prediction of training data

training_accuracy_knn = metrics.accuracy_score(y, result_knn)	
training_accuracy_svm = metrics.accuracy_score(y, result_svm)	
#training accuracy for nearest neighbors
c_matrix_knn = metrics.confusion_matrix(y, result_knn)		
c_matrix_svm = metrics.confusion_matrix(y, result_svm)		
#confusion matrix
c_matrix_svm = metrics.confusion_matrix(y, result_svm)	#confusion matrix
#print "kNN training accuracy:",training_accuracy_knn
#print "SVM training accuracy:",training_accuracy_svm
#print "Confusion matrix for knn:\n", c_matrix_knn
#print "Confusion matrix for svm:\n", c_matrix_svm
