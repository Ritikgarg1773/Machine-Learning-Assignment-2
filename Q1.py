import numpy as np
import h5py
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

choice = input("Enter 1 to use PCA and 2 for SVD and 3 for stratified sampling: ")
if(choice =='1'):
	print("--------------------------PCA--------------------------------------")
	data = h5py.File('part_A_train.h5','r')
	# data = h5py.File('/content/drive/My Drive/Colab/datafiles/part_B_train.h5','r')
	# print(data)
	X = data['X']
	X = pd.DataFrame(X[()])
	pca = PCA(n_components=10)
	X_pca = pca.fit_transform(X)
	# print(X_pca)
	# print(X)
	Y = data['Y']
	# print(Y)
	# X_test = pd.DataFrame(X_pca.value)
	# print(X_pca,X_pca.shape)
	y = pd.DataFrame(Y[()])
	y = 0*(y[0]) + 1*(y[1]) + 2*(y[2]) + 3*(y[3]) + 4*(y[4]) + 5*(y[5]) + 6*(y[6]) + 7*(y[7]) + 8*(y[8]) + 9*(y[9])  #converted y in single column
	# print(y.shape)
	a = int(X_pca.shape[0]*0.8)
	X_pca_train = X_pca[:a]
	X_pca_test = X_pca[a:]
	y_train = np.array(y[:a])
	y_test = np.array(y[a:])
	# print(X_pca_train.shape,y_train.shape)
	choice1 = input("Enter 1 to view the accuracy, 2 for tSNE: ")
	if(choice1=='1'):
		from sklearn.linear_model import LogisticRegression
		from sklearn import metrics 

		logistic = LogisticRegression()
		logistic.fit(X_pca_train,y_train)
		y_pred = logistic.predict(X_pca_test)
		# print(y_pred)
		accu = metrics.accuracy_score(y_pred,y_test)
		print("Accuracy using PCA is: ",accu)
	elif(choice1 =='2'):
		# import numpy as np
		# import h5py
		# import pandas as pd
		# from sklearn.decomposition import PCA
		"""Source: Visualising high-dimensional datasets using PCA and t-SNE in Python"""
		from sklearn.manifold import TSNE
		import seaborn as sns
		import matplotlib.pyplot as plt

		# data = h5py.File('part_A_train.h5','r')
		# # data = h5py.File('/content/drive/My Drive/Colab/datafiles/part_B_train.h5','r')
		# # print(data)
		# X = data['X']
		# X = pd.DataFrame(X[()])
		pca = PCA(n_components=3)
		X_pca = pca.fit_transform(X)
		X['pca-one'] = X_pca[:,0]
		X['pca-two'] = X_pca[:,1] 
		X['pca-three'] = X_pca[:,2]
		tSNE = TSNE(n_components=2,random_state=0)
		X_tSNE = tSNE.fit_transform(X_pca)
		# print(X_pca)
		# print(X)
		Y = data['Y']
		# print(Y)
		# X_test = pd.DataFrame(X_pca.value)
		# print(X_svd,X_svd.shape)
		y = pd.DataFrame(Y[()])
		y = 0*(y[0]) + 1*(y[1]) + 2*(y[2]) + 3*(y[3]) + 4*(y[4]) + 5*(y[5]) + 6*(y[6]) + 7*(y[7]) + 8*(y[8]) + 9*(y[9])  #converted y in single column
		# print(y.shape)
		X['tsne-pca50-one'] = X_tSNE[:,0]
		X['tsne-pca50-two'] = X_tSNE[:,1]
		X['Y'] = y
		plt.figure(figsize=(16,4))
		ax1 = plt.subplot(1, 3, 1)       #plotting the scatterplots to show the segregation of the datapoints
		sns.scatterplot(
		    x="pca-one", y="pca-two",
		    hue="Y",
		    palette=sns.color_palette("hls", 10),
		    data=X,
		    legend="full",
		    alpha=0.3,
		    ax=ax1
		)
		ax3 = plt.subplot(1, 3, 3)
		sns.scatterplot(
		    x="tsne-pca50-one", y="tsne-pca50-two",
		    hue="Y",
		    palette=sns.color_palette("hls", 10),
		    data=X,
		    legend="full",
		    alpha=0.3,
		    ax=ax3
		)
		plt.show()
	else:
		print("Invalid choice")
elif(choice == '2'):
	print("--------------------------SVD--------------------------------------")
	data = h5py.File('part_A_train.h5','r')
	# data = h5py.File('/content/drive/My Drive/Colab/datafiles/part_B_train.h5','r')
	# print(data)
	X = data['X']
	X = pd.DataFrame(X[()])
	svd = TruncatedSVD(n_components=10)
	X_svd = svd.fit_transform(X)
	# print(X_pca)
	# print(X)
	Y = data['Y']
	# print(Y)
	# X_test = pd.DataFrame(X_pca.value)
	# print(X_svd,X_svd.shape)
	y = pd.DataFrame(Y[()])
	y = 0*(y[0]) + 1*(y[1]) + 2*(y[2]) + 3*(y[3]) + 4*(y[4]) + 5*(y[5]) + 6*(y[6]) + 7*(y[7]) + 8*(y[8]) + 9*(y[9])  #converted y in single column
	# print(y.shape)
	a = int(X_svd.shape[0]*0.8)
	X_svd_train = X_svd[:a]
	X_svd_test = X_svd[a:]
	y_train = np.array(y[:a])
	y_test = np.array(y[a:])
	# print(X_svd_train.shape,y_train.shape)
	choice1 = input("Enter 1 to view the accuracy and 2 for tSNE: ")
	if(choice1 == '1'):
		from sklearn.linear_model import LogisticRegression
		from sklearn import metrics 

		logistic = LogisticRegression()
		logistic.fit(X_svd_train,y_train)
		y_pred = logistic.predict(X_svd_test)
		# print(y_pred)
		accu = metrics.accuracy_score(y_pred,y_test)
		print("Accuracy for svd is: ",accu)
	elif(choice1 == '2'):
		from sklearn.manifold import TSNE
		import seaborn as sns
		import matplotlib.pyplot as plt

		# data = h5py.File('/content/gdrive/My Drive/Colab/datafiles/part_A_train.h5','r')
		# # data = h5py.File('/content/drive/My Drive/Colab/datafiles/part_B_train.h5','r')
		# # print(data)
		# X = data['X']
		# X = pd.DataFrame(X[()])
		svd = TruncatedSVD(n_components=10) #applying SVD to reduce the dimensions
		X_svd = svd.fit_transform(X)
		X['svd-one'] = X_svd[:,0]
		X['svd-two'] = X_svd[:,1] 
		X['svd-three'] = X_svd[:,2]
		print("reduction using svd: ",X_svd.shape)
		tSNE = TSNE(n_components=2,random_state=0) #Now applying tSNE to reduce it further
		X_tSNE = tSNE.fit_transform(X_svd)
		# print(X_pca)
		# print(X)
		Y = data['Y']
		# print(Y)
		# X_test = pd.DataFrame(X_pca.value)
		print("using tSNE: ",X_tSNE.shape)
		y = pd.DataFrame(Y[()])
		y = 0*(y[0]) + 1*(y[1]) + 2*(y[2]) + 3*(y[3]) + 4*(y[4]) + 5*(y[5]) + 6*(y[6]) + 7*(y[7]) + 8*(y[8]) + 9*(y[9])  #converted y in single column
		# print(y.shape)
		X['tsne-svd-one'] = X_tSNE[:,0]
		X['tsne-svd-two'] = X_tSNE[:,1]
		X['Y'] = y
		plt.figure(figsize=(16,4))
		ax1 = plt.subplot(1, 3, 1)
		sns.scatterplot(                             #scatter plots to show the segregation of the dataplots
		    x="svd-one", y="svd-two",
		    hue="Y",
		    palette=sns.color_palette("hls", 10),
		    data=X,
		    legend="full",
		    alpha=0.3,
		    ax=ax1
		)
		ax3 = plt.subplot(1, 3, 3)
		sns.scatterplot(
		    x="tsne-svd-one", y="tsne-svd-two",
		    hue="Y",
		    palette=sns.color_palette("hls", 10),
		    data=X,
		    legend="full",
		    alpha=0.3,
		    ax=ax3
		)
		plt.show()
	else:
		print("Invalid choice")
elif(choice == '3'):
	print("----------------------------------------Stratified sampling---------------------------------------")
	# import numpy as np
	from sklearn.model_selection import StratifiedShuffleSplit
	# import h5py
	# import pandas as pd
	import matplotlib.pyplot as plt

	data = h5py.File('part_A_train.h5','r')
	# print(data)
	X = data['X']
	X = pd.DataFrame(X[()])
	X=np.array(X)
	Y = data['Y']
	y = pd.DataFrame(Y[()])
	y = 0*(y[0]) + 1*(y[1]) + 2*(y[2]) + 3*(y[3]) + 4*(y[4]) + 5*(y[5]) + 6*(y[6]) + 7*(y[7]) + 8*(y[8]) + 9*(y[9])  #converted y in single column
	y= np.array(y)
	# print(y)
	Stratified_sample = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
	# Stratified_sample.get_n_splits(X, y)
	fig, (ax1) = plt.subplots(1, figsize=(11,4))
	# ax1.hist(X)
	# ax1.set_xlabel('X')
	ax1.hist(y)                     #generated the histogram to show the class distribution
	ax1.set_xlabel('y');
	plt.show()
	for train_index, test_index in Stratified_sample.split(X,y):
	  # print("TRAIN:", train_index, "TEST:", test_index)
	  X_train, X_test = X[train_index], X[test_index]
	  y_train, y_test = y[train_index], y[test_index]
	  # plt.clear()
	  fig, (ax1, ax2) = plt.subplots(1,2, figsize=(11,4))
	  ax1.hist(y_test)
	  ax1.set_xlabel('y_test')
	  ax2.hist(y_train)
	  ax2.set_xlabel('y_train');
	  plt.show()
else:
	print("Invalid choice")