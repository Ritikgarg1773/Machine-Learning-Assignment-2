import numpy as np

class Gaussian_Naivebayes:
  ''' Source: Naive Bayes in Python - Machine Learning From Scratch 05 - Python Tutorial'''
  def fit(self,X,y):
    '''X is a numpy nd array with first dimensions as the number of samples and second as number of features'''
    n_samples,n_features = X.shape
    self._classes = np.unique(y) #array of all unique values in y
    # print(self._classes.shape)
    n_classes = (self._classes.shape[0])
    # print(n_samples,n_features)
    # print(n_classes)
    #init mean, var, priors
    self._mean = np.zeros((n_classes,n_features),dtype=np.float64)
    self._var = np.zeros((n_classes,n_features),dtype=np.float64)
    self._priors = np.zeros(n_classes,dtype=np.float64)
    for c in self._classes:
      X_c = X[c==y]
      # c is of float type so convert it into int
      self._mean[int(c),:] = X_c.mean(axis=0)
      self._var[int(c),:] = X_c.var(axis=0)
      self._priors[int(c)] = X_c.shape[0]/float(n_samples) #NUMBER OF  SAMPLES WITH THE LABEL

  def predict(self,X):
    y_pred = [self._predict(x) for x in X]
    return y_pred      #returning the predicted values

  def _predict(self,x):
    ''' calculate the class ,posterior and prior and choose the class with highest prob'''
    posteriors = []
    for index, c in enumerate(self._classes):  #with enumerate we get the index and class label
      prior = np.log(self._priors[index])
      class_conditional = np.sum(np.log(self._pdf(index,x)))
      posterior = prior + class_conditional
      posteriors.append(posterior)
    return self._classes[np.argmax(posteriors)] #choose the class with highest probability

  def _pdf(self,class_index, x):
    mean = self._mean[class_index]
    var = self._var[class_index]
    for c in range((var.shape[0])): #if any value of var is 0, convert it to 1e6, to remove divide by 0 error
      if(var[c]==0):
        var[c]=10**(-4)
    numerator = np.exp(-1*(x-mean)**2/(2*var))
    denominator = np.sqrt(2*np.pi*var)
    # print(numerator/denominator)
    answer = numerator/denominator
    for c in range((answer.shape[0])): #if any value of answer is 0, convert it to 1e6, to remove divide by 0 error or invalid input error
      if(answer[c]==0):
        answer[c]=10**(-4)
    return answer 