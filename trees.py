import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble, metrics, model_selection
import pdb

data = np.genfromtxt('plrx.txt')

X = data[:,:12]
y = data[:, -1]
scores = []							#array that will contain the accuracy scores for each classification run

def trees(X,y):							#random forest classifier defined as a function
	clf = ensemble.RandomForestClassifier(n_estimators=10,)
	clf.fit(X,y)
	results = clf.predict(X)
	return results;

def plot_roc(tpr,fpr,thresholds):
   fig = plt.gcf()
   fig.set_size_inches(5.5, 4.5)
   plt.plot(fpr, tpr,label='correct')
   plt.plot([0,1],[0,1],'r--')
   plt.title('ROC curve for Caffe-based classifier')
   plt.ylabel('True Positive Rate')
   plt.xlabel('False Positive Rate')
   plt.legend(loc='best')
   plt.grid()
#   plt.savefig('roc_plot.pgf')
   plt.show()

for x in range(100):						#'for' loop to run the clf for 100 iterations
	accuracy = metrics.accuracy_score(y, trees(X,y))
	scores.append(accuracy)


clf = ensemble.RandomForestClassifier(n_estimators=30,max_depth=3)
loo_scores= []
probs=[]
loo = model_selection.LeaveOneOut()
for train, test in loo.split(X):
    clf.fit(X[train], y[train])
    acc = clf.score(X[test], y[test])
    temp_pred=clf.predict_proba(X[test])
    probs.append(temp_pred[0,0])
    loo_scores.append(acc)
fpr, tpr, thresholds = metrics.roc_curve(y ,probs, pos_label= 2)
#print 100*np.mean(loo_scores)
#print np.shape(loo_scores), np.shape(y)
plot_roc(tpr,fpr,thresholds)
print "AUC: ",metrics.roc_auc_score(y,probs)
