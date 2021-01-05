import numpy as np
import h5py
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import Gaussian_NB as gnb


d1 = input("Enter 1 to view dataset A and 2 for dataset b: ")
if(d1 == '1'):
	data = h5py.File('part_A_train.h5','r')
	# data = h5py.File('/content/drive/My Drive/Colab/datafiles/part_B_train.h5','r')
	# print(data)
	# for key in data.keys():
	#     print(key)
	X = data['X']
	# print(X)
	Y = data['Y']
	# print(Y)

	x = pd.DataFrame(X[()])  #taken the value of X
	# print(x.head())
	y = pd.DataFrame(Y[()]) #taken the value of y
	# print(y)
	y = 0*(y[0]) + 1*(y[1]) + 2*(y[2]) + 3*(y[3]) + 4*(y[4]) + 5*(y[5]) + 6*(y[6]) + 7*(y[7]) + 8*(y[8]) + 9*(y[9])  #converted y in single column
	# print(y.head())
	# y=0*(y[0])
	# print(x.shape)
	# print(y.shape)
	x = np.array(x)
	len = x.shape[0]
	a = int(0.8*len)
	X_train = x[:a]
	X_test = x[a:]
	print("Shape of X_train and X_test: ",(X_train.shape),(X_test.shape))
	y = np.array(y)
	y_test = y[a:]
	y_train = y[:a]
	print("shape of y_train, y_test: ",(y_train.shape),(y_test.shape))

	a =  input("Enter 1 to view results using sklearn and 2 for implementation from scratch: ")
	if(a == '1'):
		clf = GaussianNB()
		model = clf.fit(x,y)
		y_pred = clf.predict(x)
		accu = metrics.accuracy_score(y_pred,y)
		print(accu)
	elif(a=='2'):
		def accuracy(y_true,y_pred):
		  accuracy = np.sum(y_true == y_pred)/ (y_true.shape[0])
		  return accuracy
		nb = gnb.Gaussian_Naivebayes()
		nb.fit(X_train,y_train)
		y_pred = nb.predict(X_test)
		print("accuracy: ", accuracy(y_test,y_pred))
	else :
		print("Invalid input")
elif (d1 == '2'):
	data = h5py.File('part_B_train.h5','r')
	# data = h5py.File('/content/drive/My Drive/Colab/datafiles/part_B_train.h5','r')
	print(data)
	# for key in data.keys():
	#     print(key)
	X = data['X']
	# print(X)
	Y = data['Y']
	# print(Y)
	x = pd.DataFrame(X[()])
	print(x.head())
	y = pd.DataFrame(Y[()])
	# print(y.head())
	y = 0*(y[0]) + 1*(y[1])
	print(y.head(),y.shape)
	x = np.array(x)
	len = x.shape[0]
	a = int(0.8*len)
	X_train = x[:a]
	X_test = x[a:]
	print("Shape of X_train and X_test: ",(X_train.shape),(X_test.shape))
	y = np.array(y)
	y_test = y[a:]
	y_train = y[:a]
	print("shape of y_train, y_test: ",(y_train.shape),(y_test.shape))

	a =  input("Enter 1 to view results using sklearn and 2 for implementation from scratch: ")
	if(a == '1'):
		clf = GaussianNB()
		model = clf.fit(x,y)
		y_pred = clf.predict(x)
		accu = metrics.accuracy_score(y_pred,y)
		print(accu)
	elif(a=='2'):
		def accuracy(y_true,y_pred):
		  accuracy = np.sum(y_true == y_pred)/ (y_true.shape[0])
		  return accuracy
		nb = gnb.Gaussian_Naivebayes()
		nb.fit(X_train,y_train)
		y_pred = nb.predict(X_test)
		print("accuracy: ", accuracy(y_test,y_pred))
	else :
		print("Invalid input")
else:
	print("Invalid Input")