import numpy as np
from sklearn import ensemble, metrics

data = np.genfromtxt('plrx.txt')

X = data[:,:12]
y = data[:, -1]

clf = ensemble.RandomForestClassifier(n_estimators=10)
clf = clf.fit(X,y)

results = clf.predict(X)

accuracy = metrics.accuracy_score(y, results)
print accuracy

