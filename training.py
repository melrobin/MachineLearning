import numpy as np
from sklearn import neighbors, metrics, svm, ensemble, model_selection

raw_data = np.genfromtxt('plrx.txt')	#plrx.txt is the dataset txt file 

data = raw_data[:,:12]			#slicing the raw data
label = raw_data[:,-1]
X = data 				#data for testing
y = label 

clf1 = neighbors.KNeighborsClassifier(n_neighbors=2)
clf2 = svm.SVC(gamma = 5)
clf3 = ensemble.RandomForestClassifier(n_estimators = 30, max_depth= 3)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size= 0.5)

svm_score = []
trees_score = []

for i in range(10):
	clf2.fit(X_train,y_train)
	clf3.fit(X_train,y_train)
	score2 = clf2.score(X_test, y_test)
	score3 = clf3.score(X_test, y_test)
	svm_score.append(score2)
	trees_score.append(score3)

#print "SVM:", 100*np.mean(svm_score)
#print metrics.confusion_matrix(y_test, clf2.predict(X_test))
#print "RF:", 100*np.mean(trees_score)
#print metrics.confusion_matrix(y_test, clf3.predict(X_test))

clf1.fit(X,y)
clf2.fit(X,y)
clf3.fit(X,y)

knn_tscore = metrics.accuracy_score(y, clf1.predict(X))
svm_tscore = metrics.accuracy_score(y, clf2.predict(X))
rf_tscore = metrics.accuracy_score(y, clf3.predict(X))

print "knn training:", 100*knn_tscore
print "SVM training:", 100*svm_tscore
print "rf training:", 100*rf_tscore
