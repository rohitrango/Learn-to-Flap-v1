from sklearn import svm
import numpy as np
from pykeyboard import PyKeyboard
X = np.loadtxt('train.csv',delimiter=",")
X = np.random.shuffle(X)
m, n = X.shape
X_train = X[0:int(3*m/4),:]
y_train = X_train[0:int(3*m/4),n-1]
X_test = X[int(3*m/4):,:]
y_test = X_train[int(3*m/4):,n-1]
X_train,X_test = X_train[:,0:n-1],X_test[:,0:n-1]
clf = svm.LinearSVC()
clf.fit(X_train, y_train)
y = clf.predict(X_test)
print(sum(abs(y-y_test)))