import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import random
'''Source: Lecture slides'''
'''------------------------------------------Data Preprocessing-----------------------------------------------------------------'''
data = pd.read_csv('weight-height.csv',sep=",")
X = data.iloc[:,1]
y=data.iloc[:,2]
a = int(len(X)*0.8)
X_train = X[:a] #chosen for model fit
X_test = X[a:]  #created the bagging points
y_train = y[:a] #for model fit
y_test = y[a:]  #for bagging

B = 500#Bagging size = 500

y_sample = np.zeros((len(X_test),B)) #created an array of size(len(X_test),500)

for i in range(B):
  '''choose random 9000 samples from X_train adn perform model.fit on them'''
  bootstrap_samples = np.random.choice(np.random.randint(0,a,a),size = a)
  X_bootstrap_sample = np.array(X_train[bootstrap_samples]).reshape(-1,1)
  y_bootstrap_sample = np.array(y_train[bootstrap_samples])
  linear = linear_model.LinearRegression()
  linear.fit(X_bootstrap_sample,y_bootstrap_sample)
  y_bootstrap = linear.predict(np.array(X_test).reshape(-1,1))
  y_sample[:,i] = y_bootstrap
'''  ----------------------------- Calculating the Mean, Bias, Varience and the MSE-----------------------------------------'''
Mean =  np.sum(y_sample,axis=1)/B #found the mean
print(Mean," Mean vector")
print(Mean@Mean.T, " Mean")
Bias = Mean - np.array(y_test) #found the bias
print(Bias, " Bias Vector")
print(Bias@Bias.T," Bias")
Varience = np.sum((y_sample - Mean.reshape(-1,1))**2,axis=1)/(B-1)  #found the varience
print(Varience, " varience vector")
print(Varience@Varience.T," Varience")
MSE = np.sum((y_sample - np.array(y_test).reshape(-1,1))**2,axis=1)/B #found the MSE
#Validating the formula given
answer = MSE - (Bias**2) - Varience 
print("Value of MSE - Bias**2 - Varience: ",np.dot(answer.T,answer)) #it comes out to be very small, that will be the noise