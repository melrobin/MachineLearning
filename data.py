import numpy as np
from sklearn import neighbors, model_selection

raw_data = np.genfromtxt('plrx.txt')	#plrx.txt is the dataset txt file 

data = raw_data[:,:12]			#splicing the raw data
label = raw_data[:,-1]


X = data 				#data for testing
y = label 

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size = 0.1)

k_values = list(range(1,41))
cv_scores =[]

for k in k_values:
	knn = neighbors.KNeighborsClassifier(n_neighbors=k)
	scores = model_selection.cross_val_score(knn, X_train, y_train,cv=10, scoring = 'accuracy')
	cv_scores.append(scores.mean())

error = [1 - x for x in cv_scores]

optimal_k = error.index(min(error))
optimal_k = optimal_k +1
print optimal_k
print error

### use knn 
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size = 0.1)
knn = neighbors.KNeighborsClassifier(n_neighbors= optimal_k)
knn.fit(X_train, y_train)

accuracy = knn.score(X_test, y_test)
print accuracy
