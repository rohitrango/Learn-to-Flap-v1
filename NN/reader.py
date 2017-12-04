import csv
fo = open('train.csv','r')
of = open('train_new.csv','w')
writer = csv.writer(of)
for j in range(1000000):
	fo.readline()
for i in range(1000000):
	line = fo.readline()
	line = line.split(',')
	y = line[9]
	X = line[0:9]
	X = list(map(lambda x:int(x),X))
	X = X[0:3]
	y = int(y[0])
	if y == 0:
		X.append(1)
		X.append(0)
	else:
		X.append(0)
		X.append(1)
	writer.writerow((X))
of.close()
fo.close()