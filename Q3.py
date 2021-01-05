import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
# import 
Dataset_choice = input("Enter 1 for Dataset A and 2 for Dataset B: ")
# data = h5py.File('part_A_train.h5','r')
if(Dataset_choice=='1'):
	data = h5py.File('part_A_train.h5','r')
	# print(data)
	X = data['X']
	X = pd.DataFrame(X[()])
	X=np.array(X)
	Y = data['Y']
	y = pd.DataFrame(Y[()])
	y = 0*(y[0]) + 1*(y[1]) + 2*(y[2]) + 3*(y[3]) + 4*(y[4]) + 5*(y[5]) + 6*(y[6]) + 7*(y[7]) + 8*(y[8]) + 9*(y[9])  #converted y in single column
	# y = 0*(y[0]) + 1*(y[1])  #converted y in single column
	y= np.array(y)

	''' Divide the data into test-val-split as 60-20-20. For that I divided the data into 80-20 train and test.
	    In test data, I will apply 4 KFoldCV so data will become 60-20-20 for test-val-split'''
	a = len(X)
	# print(a)
	X_train = X[:int(a*0.8)]
	X_test = X[int(a*0.8):]
	y_train = y[:int(a*0.8)]
	y_test = y[int(a*0.8):]
	# print(X_train.shape,X_test.shape)
	# print(y_train.shape,y_test.shape)
elif(Dataset_choice=='2'):
	data = h5py.File('part_B_train.h5','r')
	# print(data)
	X = data['X']
	X = pd.DataFrame(X[()])
	X=np.array(X)
	Y = data['Y']
	y = pd.DataFrame(Y[()])
	# y = 0*(y[0]) + 1*(y[1]) + 2*(y[2]) + 3*(y[3]) + 4*(y[4]) + 5*(y[5]) + 6*(y[6]) + 7*(y[7]) + 8*(y[8]) + 9*(y[9])  #converted y in single column
	y = 0*(y[0]) + 1*(y[1])  #converted y in single column
	y= np.array(y)

	''' Divide the data into test-val-split as 60-20-20. For that I divided the data into 80-20 train and test.
	    In test data, I will apply 4 KFoldCV so data will become 60-20-20 for test-val-split'''
	a = len(X)
	# print(a)
	X_train = X[:int(a*0.8)]
	X_test = X[int(a*0.8):]
	y_train = y[:int(a*0.8)]
	y_test = y[int(a*0.8):]
	# print(X_train.shape,X_test.shape)
	# print(y_train.shape,y_test.shape)
else:
	print("Invalid choice")
Model = input("Enter 1 for Decision Tree and 2 for Gaussian NB: ")
if(Model == '1'):
	print("Decision Tree")
	print("Applying 4 Fold CV to find the optimal depth of the decision tree with maximum accuracy -> Preforming GridSearch CV")
	import pickle
	import matplotlib.pyplot as plt
	from sklearn import tree
	# from sklearn.metrics import accuracy_score
	'''Source: Geeks for Geeks: pickle'''
	def accuracy(y_true,y_pred):
	  accuracy = np.sum(y_true == y_pred)/ (y_true.shape[0])
	  return accuracy

	''' K fold from scratch for Decision Trees'''
	# depth = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
	accuracies_val= []
	accuracies_train = []
	max_accuracy=-999
	max_depth = 1
	k = 4                                                        #Creating 4 folds in CV
	X_folds = np.array_split(X_train,k)
	y_folds = np.array_split(y_train,k)
	for d in range(1,50):
	  accu_fold_val=0
	  accu_fold_train = 0
	  for i in range(k):
	    X_data_train = X_folds.copy()                             # Creating the test and validation data
	    y_data_train = y_folds.copy()
	    X_data_test = X_folds[i]
	    y_data_test = y_folds[i]
	    del X_data_train[i]
	    del y_data_train[i]
	    X_data_train = np.concatenate(X_data_train)
	    y_data_train = np.concatenate(y_data_train)  #done with data generation train and validate
	    # print(len(X_data_test),len(X_data_train))
	    '''apply Decision tree on train data and get the accuracy using a depth'''
	    dt = tree.DecisionTreeClassifier(criterion='gini',max_depth=d)
	    dt.fit(X_data_train,y_data_train)
	    y_pred_val = dt.predict(X_data_test)
	    accu_val = accuracy(y_data_test,y_pred_val)             #prediciting accuracy on the 
	    y_pred_train = dt.predict(X_data_train)
	    accu_train = accuracy(y_data_train,y_pred_train)
	    # accu = accuracy(y_data_test,y_pred)
	    # print(accu)
	    if(accu_val>max_accuracy):
	      max_accuracy = accu_val
	      max_depth = d
	      saved_model = pickle.dumps(dt)
	    accu_fold_val+=(accu_val)
	    accu_fold_train+=(accu_train)
	  accuracies_val.append(accu_fold_val/k)
	  accuracies_train.append(accu_fold_train/k)
	plt.plot(accuracies_val,'o-',color="r",
	             label="Validation score")
	plt.plot(accuracies_train,'o-',color="g",
	             label="Train score")
	plt.show()
	print(accuracies_val,"Accuracy on validation data using Decision Tree")
	print("Maximum accuracy and max depth: ",max_accuracy, max_depth)
	best_model = pickle.loads(saved_model) #loading the best model to predict on test data
	print(best_model.get_depth(),": Model depth equal to the max depth")
	y_test_pred = best_model.predict(X_test)
	accur_test = accuracy(y_test,y_test_pred)
	print("Test accuracy uding the best model: ",accur_test)
elif(Model == '2'):
	''' Using Gaussian Naive bayes to train the data'''
	'''Source: Geeks for Geeks: pickle'''
	from sklearn.naive_bayes import GaussianNB
	import pickle
	import matplotlib.pyplot as plt
	# from sklearn import metrics 

	def accuracy(y_true,y_pred):
	  accuracy = np.sum(y_true == y_pred)/ (y_true.shape[0])
	  return accuracy

	accuracies_val= []
	accuracies_train = []
	max_accuracy=-999
	# max_depth = 1
	k = 4  #Creating 4 folds in CV
	X_folds = np.array_split(X_train,k)
	y_folds = np.array_split(y_train,k)
	accu_fold_val=0
	accu_fold_train = 0
	for i in range(k):
	  X_data_train = X_folds.copy()
	  y_data_train = y_folds.copy()
	  X_data_test = X_folds[i]
	  y_data_test = y_folds[i]
	  del X_data_train[i]
	  del y_data_train[i]
	  X_data_train = np.concatenate(X_data_train)
	  y_data_train = np.concatenate(y_data_train)  #done with data generation train and validate
	  # print(len(X_data_test),len(X_data_train))
	  #apply Decision tree on train data and get the accuracy using a depth
	  clf = GaussianNB()
	  model = clf.fit(X_data_train,y_data_train)
	  # y_pred = clf.predict(X_test)
	  y_pred_val = clf.predict(X_data_test)
	  accu_val = accuracy(y_data_test,y_pred_val)
	  y_pred_train = clf.predict(X_data_train)
	  accu_train = accuracy(y_data_train,y_pred_train)
	  # accu = accuracy(y_data_test,y_pred)
	  # print(accu)
	  if(accu_val>max_accuracy):
	    max_accuracy = accu_val
	    # max_depth = d
	    saved_model = pickle.dumps(clf)
	  accu_fold_val+=(accu_val)
	  accu_fold_train+=(accu_train)
	accuracies_val.append(accu_fold_val/k)
	accuracies_train.append(accu_fold_train/k)
	# plt.plot(accuracies_val,'o-',color="r",
	            # label="Validation score")
	# plt.plot(accuracies_train,'o-',color="g",
	            # label="Train score")
	print("Accuracy on validate: ",accuracies_val)
	print("Maximum Accuracy: ",max_accuracy)
	best_model = pickle.loads(saved_model) #loading the best model to predict on test data
	# print(best_model.get_depth(),"Model depth equal to the max depth")
	y_test_pred = best_model.predict(X_test)
	accur_test = accuracy(y_test,y_test_pred)
	print("Test accuracy using GaussianNB: ",accur_test)
else:
	print("Invalid choice")

def evaluation_metric(y_test,y_pred,class_type):
  '''Calculate Accuracy, Precision, Recall, F1_score, plot ROC, 
      Returns Confusion matrix. Type is 1 for binary class and 2 for multiclass
      Source: https://www.python-course.eu/confusion_matrix.php, '''
  #Confusion matrix for multiclass and binary class
  n_classes = len(np.unique(y_test))
  # print(n_classes)
  confusion_matrix = np.array([[0] * n_classes for i in range(n_classes)])
  for pred, exp in zip((y_pred), (y_test)):
      confusion_matrix[int(exp)][int(pred)] += 1
  if(class_type==1):
    '''Calculating accuracy: (TP+TN)/(TP+FN+FP+TN)'''
    acc_denominator = sum(sum(l) for l in confusion_matrix) #total sum of the matrix
    acc_numerator = sum(confusion_matrix[i][i] for i in range(len(confusion_matrix)))
    accuracy = acc_numerator/acc_denominator 
    print("Accuracy of the data: ",accuracy)
    '''Calculating precision: TP/(TP+FP)'''
    prec_numerator = confusion_matrix[1][1]
    prec_denominator = sum(confusion_matrix[i][1] for i in range(len(confusion_matrix)))
    precision = prec_numerator/prec_denominator
    print("Precision on the data: ",precision)
    '''Calculating recall: (TP)/(TP+FN)'''
    rec_numerator=confusion_matrix[1][1]
    rec_denominator= sum(confusion_matrix[1][i] for i in range(len(confusion_matrix)))
    recall = rec_numerator/rec_denominator
    print("Recall on the data: ",recall)
    '''Calculating f1_score'''
    f1_score = (2*precision*recall)/(precision+recall)
    print("F1_score on the data: ",f1_score)
  elif(class_type==2):
    '''Accuracy'''
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    accuracy = diagonal_sum / sum_of_all_elements
    print("Accuracy on the data: ",accuracy)
    '''Precision AND recall for individual class'''
    for i in range(n_classes):
      column_sum = confusion_matrix[:, i]
      precision = np.nan_to_num(confusion_matrix[i, i] / column_sum.sum())
      row_sum = confusion_matrix[i, :]
      recall =  np.nan_to_num(confusion_matrix[i, i] / row_sum.sum())
      print("Printing the precision and recall of individual classes: ")
      print(f"{i:5d} {precision:9.3f} {recall:6.3f}")

    """Macro Average for precision"""
    rows, columns = confusion_matrix.shape
    sum_of_precisions = 0
    for i in range(rows):
      column_sum = confusion_matrix[:, i]
      precision = np.nan_to_num(confusion_matrix[i, i] / column_sum.sum())
      sum_of_precisions += precision
    precision_macro =  np.nan_to_num(sum_of_precisions / rows)
    print("Precision_macro_value: ",precision_macro)
    """Macro Average for recall"""
    rows, columns = confusion_matrix.shape
    sum_of_recalls = 0
    for i in range(columns):
      row_sum = confusion_matrix[i, :]
      recall =  np.nan_to_num(confusion_matrix[i, i] / row_sum.sum())
      sum_of_recalls += recall
    recall_macro= np.nan_to_num(sum_of_recalls / columns)
    print("Recall_macro_value: ",recall_macro)
    '''Macro Average F1_Score'''
    f1_score_macro = np.nan_to_num(2*recall_macro*precision_macro)/np.nan_to_num(recall_macro+precision_macro)
    print("F1_score_ macro: ",f1_score_macro)
  else:
    print("invalid choice")
  return confusion_matrix

print()
print("---------------------------Printing the confusion_matrix--------------------------------------")

confusion_matrix = evaluation_metric(y_test,y_test_pred,int(Dataset_choice))
print("confusion_matrix: ",confusion_matrix)

print()
print("----------------------ROC Curves------------------------------------------")
def roc_curve(y, prob):
    '''Source: https://towardsdatascience.com/roc-curve-a-complete-introduction-2f2da2e0434c'''
    tpr_list = []
    fpr_list = []
    threshold = [0.0 + x*(1.1-0.0)/10 for x in range(11)]
    for t in threshold:
        y_pred = np.zeros(y.shape[0])
        if(Dataset_choice=='1'):
            y_pred[prob[:,i] >= t] = 1
        else:
            y_pred[prob[:,1] >= t] = 1
        TN = len(y_pred[(y_pred == y) & (y == 0)])
        TP = len(y_pred[(y_pred == y) & (y == 1)])
        FP = len(y_pred[(y_pred != y) & (y == 0)])
        FN = len(y_pred[(y_pred != y) & (y == 1)])
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        tpr_list.append(TPR)
        fpr_list.append(FPR)
    return fpr_list, tpr_list, threshold

prob = best_model.predict_proba(X_test)
print("Predict_proba: ",prob)
fpr, tpr, threshold = roc_curve(y_test, prob)

plt.plot(fpr, tpr, 'b')
plt.plot([0,1],[0,1], 'r--')
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)

plt.show()