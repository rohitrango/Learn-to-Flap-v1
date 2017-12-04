from sklearn import svm
import numpy as np
from pykeyboard import PyKeyboard
from sklearn.metrics import f1_score
X = np.loadtxt('train.csv',delimiter=",")
np.random.shuffle(X)
m, n = X.shape
X_train = X[0:int(3*m/4),0:n-1]
y_train = X[0:int(3*m/4),n-1]
X_test = X[int(3*m/4):m,0:n-1]
y_test = X[int(3*m/4):m,n-1]
clf = svm.LinearSVC()
clf.fit(X_train, y_train)
y = clf.predict(X_test)
print(f1_score(y,y_test,average='micro'))