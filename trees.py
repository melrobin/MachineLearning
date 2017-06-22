import numpy as np
from sklearn import ensemble, metrics

data = np.genfromtxt('plrx.txt')

X = data[:,:12]
y = data[:, -1]
scores = []							#array that will contain the accuracy scores for each classification run

def trees(X,y):							#random forest classifier defined as a function
	clf = ensemble.RandomForestClassifier(n_estimators=10)
	clf.fit(X,y)
	results = clf.predict(X)
	return  results;

for x in range(100):						#'for' loop to run the clf for 100 iterations
	accuracy = metrics.accuracy_score(y, trees(X,y))
	scores.append(accuracy)

print 100*np.mean(scores)					#average of the 100 scores

