import numpy as np
from sklearn import ensemble, metrics, model_selection

data = np.genfromtxt('plrx.txt')

X = data[:,:12]
y = data[:, -1]
scores = []							#array that will contain the accuracy scores for each classification run

def trees(X,y):							#random forest classifier defined as a function
	clf = ensemble.RandomForestClassifier(n_estimators=10,)
	clf.fit(X,y)
	results = clf.predict(X)
	return results;

for x in range(100):						#'for' loop to run the clf for 100 iterations
	accuracy = metrics.accuracy_score(y, trees(X,y))
	scores.append(accuracy)


clf = ensemble.RandomForestClassifier(n_estimators=30,max_depth=3)
loo_scores= []

loo = model_selection.LeaveOneOut()
for train, test in loo.split(X):
	clf.fit(X[train], y[train])
	acc = clf.score(X[test], y[test])
	loo_scores.append(acc)

print 100*np.mean(loo_scores)
