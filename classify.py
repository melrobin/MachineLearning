import numpy as np
from sklearn import ensemble, neighbors, svm, metrics, model_selection

raw_data = np.genfromtxt('plrx.txt')					#reading data from txt file

X= raw_data[:,:12]							#splicing the raw data
y = raw_data[:,-1]

clf1 = neighbors.KNeighborsClassifier(n_neighbors= 2)			#k-NN classifier
clf2 = svm.SVC(gamma = 5)						#SVM classifier
clf3 = ensemble.RandomForestClassifier(n_estimators=30, max_depth=3)	#Random Forest classifier

#empty arrays to hold classifier accuracy scores
knn_scores1 = []
svm_scores1= []
trees_scores1 = []

#Spliting the data into training and testing data
kf = model_selection.KFold(n_splits=5 )
for train , test in kf.split(X):
	clf1.fit(X[train],y[train])
	clf2.fit(X[train],y[train])
	clf3.fit(X[train],y[train])
	acc_knn1= clf1.score(X[test],y[test])
	acc_svm1= clf2.score(X[test],y[test])
	acc_trees1= clf3.score(X[test],y[test])
	knn_scores1.append(acc_knn1)
	svm_scores1.append(acc_svm1)
	trees_scores1.append(acc_trees1)

print "\nUsing k-fold:\n", "k-NN score: ",100*np.mean(knn_scores1)
print"SVM score: ", 100*np.mean(svm_scores1)
print "Random Forest score: ", 100*np.mean(trees_scores1)

knn_scores2 = []
svm_scores2 = []
trees_scores2 = []

loo = model_selection.LeaveOneOut()
for train, test in loo.split(X):
	clf1.fit(X[train],y[train])
	clf2.fit(X[train],y[train])
	clf3.fit(X[train],y[train])
	acc_knn2 = clf1.score(X[test],y[test])
	acc_svm2 = clf2.score(X[test],y[test])
	acc_trees2 = clf3.score(X[test],y[test])
	knn_scores2.append(acc_knn2)
	svm_scores2.append(acc_svm2)
	trees_scores2.append(acc_trees2)
print "\nUsing Leave One Out:\n","k-nn score:", 100*np.mean(knn_scores2)
print "SVM score:",100*np.mean(svm_scores2)
print "Random forest score:",100*np.mean(trees_scores2)

knn_scores3 = model_selection.cross_val_score(clf1, X, y, cv=5)
svm_scores3 = model_selection.cross_val_score(clf2, X, y, cv=5)
trees_scores3 = model_selection.cross_val_score(clf3, X, y, cv=5)
print "\nUsing CV metrics:\n","k-NN score:", 100*np.mean(knn_scores3)
print "SVM score:",100*np.mean(svm_scores3)
print "Random Forest score:",100*np.mean(trees_scores3)
