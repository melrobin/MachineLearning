import numpy as np
from sklearn import ensemble, neighbors, svm, metrics, model_selection, gaussian_process
from sklearn.gaussian_process.kernels import RBF

raw_data = np.genfromtxt('plrx.txt')					#reading data from txt file

X= raw_data[:,:12]							#splicing the raw data
y = raw_data[:,-1]

clf1 = neighbors.KNeighborsClassifier(n_neighbors= 2)			#k-NN classifier
clf2 = svm.SVC(gamma = 5)						#SVM classifier
clf3 = ensemble.RandomForestClassifier(n_estimators=30, max_depth=3)	#Random Forest classifier
clf4 = gaussian_process.GaussianProcessClassifier(1*RBF([1]))		#Gaussian Process classifier

#empty arrays to hold classifier accuracy scores
knn_scores1 = []
svm_scores1= []
trees_scores1 = []
gauss_scores1=[]

#Spliting the data into training and testing data using k-Fold
kf = model_selection.KFold(n_splits=5 )				#n_splits=20 yields the best accuracy
for train , test in kf.split(X):
	clf1.fit(X[train],y[train])
	clf2.fit(X[train],y[train])
	clf3.fit(X[train],y[train])
	clf4.fit(X[train],y[train])
	acc_knn1= clf1.score(X[test],y[test])
	acc_svm1= clf2.score(X[test],y[test])
	acc_trees1= clf3.score(X[test],y[test])
	acc_gauss1= clf4.score(X[test],y[test])
	knn_scores1.append(acc_knn1)
	svm_scores1.append(acc_svm1)
	trees_scores1.append(acc_trees1)
	gauss_scores1.append(acc_gauss1)
	cf = metrics.confusion_matrix(y[test], clf2.predict(X[test]))
print cf
print "\nUsing k-fold:\n", "k-NN score: ",100*np.mean(knn_scores1)
print"SVM score: ", 100*np.mean(svm_scores1)
print "Random Forest score: ", 100*np.mean(trees_scores1)
print "Gaussian score:", 100*np.mean(gauss_scores1)

knn_scores2 = []
svm_scores2 = []
trees_scores2 = []
gauss_scores2 = []

#Spliting data using Leave One Out cross validation
loo = model_selection.LeaveOneOut()
for train, test in loo.split(X):
	clf1.fit(X[train],y[train])
	clf2.fit(X[train],y[train])
	clf3.fit(X[train],y[train])
	clf4.fit(X[train],y[train])
	acc_knn2 = clf1.score(X[test],y[test])
	acc_svm2 = clf2.score(X[test],y[test])
	acc_trees2 = clf3.score(X[test],y[test])
	acc_gauss2= clf4.score(X[test],y[test])
	knn_scores2.append(acc_knn2)
	svm_scores2.append(acc_svm2)
	trees_scores2.append(acc_trees2)
	gauss_scores2.append(acc_gauss2)

print "\nUsing Leave One Out:\n","k-nn score:", 100*np.mean(knn_scores2)
print "SVM score:",100*np.mean(svm_scores2)
print "Random forest score:",100*np.mean(trees_scores2)
print "Gaussian score:", 100*np.mean(gauss_scores2)

#Using cross validation score for the data
knn_scores3 = model_selection.cross_val_score(clf1, X, y, cv=5)		#cv=20 yields the best accuracy
svm_scores3 = model_selection.cross_val_score(clf2, X, y, cv=5)		#cv=15 yields the best accuracy
trees_scores3 = model_selection.cross_val_score(clf3, X, y, cv=5)		#cv=15 yileds the best accuracy
gauss_scores3 = model_selection.cross_val_score(clf4, X, y, cv=5)
print "\nUsing CV metrics:\n","k-NN score:", 100*np.mean(knn_scores3)
print "SVM score:",100*np.mean(svm_scores3)
print "Random Forest score:",100*np.mean(trees_scores3)
print "Gaussian score:", 100*np.mean(gauss_scores3)



