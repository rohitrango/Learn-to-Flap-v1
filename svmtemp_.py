from sklearn import svm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
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
# clf.fit(X_train, y_train)
# y = clf.predict(X_test)
# print(f1_score(y,fig = plt.figure()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
i = 0
c,m = [],[]
for x in X:
	if(i%10 == 0):
		if(x[n-1] == 0):
			c.append('r')
		else:
			c.append('b')
	i+=1
xs = X[:,1]
ys = X[:,2]
zs = X[:,0]
ax.scatter(xs, ys, zs, c=c)
plt.show()


