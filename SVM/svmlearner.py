from sklearn import svm
import numpy as np
X = np.loadtxt("train.csv",delimiter=",")
m,n = X.shape
y = X[:,n-1]
X = X[:,0:n-1]	#ignore timestamp
# print(X, X.shape)
# print(y, y.shape)
clf = svm.SVC()
clf.fit(X, y)
print("Training done.")
# s = pickle.dumps(clf,'model.pkl')
