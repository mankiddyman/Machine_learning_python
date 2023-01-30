#power is everything.
#this is the music I listend to.

#https://www.youtube.com/watch?v=TmX7wR87Hjs&t=27s
#but anyways let us begin

#the first chapter gives an example of a three class classification problem using the iris dataset and the algorithm k-nearest neighbours which works by finding the closest datapoints with the test data in dataspace and then assigning them to that datapoints' class.

#the conda environment
#conda activate py311

#meet the data
from sklearn.datasets import load_iris
import numpy as np
iris_dataset = load_iris()
#bunch object similar to dictionary is returned
print(iris_dataset.keys())

print("check this out,the dimensions\n",np.shape(iris_dataset['data']))
print("\nnow the first 10 rows of those 4 collums \n",iris_dataset['data'][:10])
print("\nthe correct classes\nthe classes as a human interpreatable string\n",iris_dataset['target'])
print("\n",iris_dataset['target_names'])
print("\n the description \n")
print(iris_dataset['DESCR'])


#now beginning properly
print("Target names:",iris_dataset['target_names'])
print("Feature names: \n",iris_dataset['feature_names'])
print("head of data :\n",iris_dataset['data'][:10])

#rows are flowers, collumns are properties 
print("Shape of the data : ",iris_dataset['data'].shape)
#individual items are called samples
#properties of a sample are its features 


#sklearn has a function to split datasets into training and test

#in sklearn data is denoted by capital X labels with lowercase y

#gonna call train_test_split on the data assign the outputs using this nomenclature.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0,test_size=0.25)
#random state is a seed 

print("X_train shape :",X_train.shape)
print("Y_train shape :",y_train.shape)
print("X_test shape :",X_test.shape)
print("Y_test shape :",y_test.shape)

import pandas as pd
#creating a dataframe from data in X_train
#labelling collumns using strings in iris_dataset.feature_names
iris_dataframe=pd.DataFrame(X_train,columns=iris_dataset.feature_names)

#creating a scatter matrix from the dataframe , color by y-train

from pandas.plotting import scatter_matrix
grr=pd.plotting.scatter_matrix(iris_dataframe,c=y_train,figsize=(15,15),marker='o',hist_kwds={'bins':20},s=60,alpha=0.8)
#you can see in some plots how they cluster apart.


#now using k-nearest-neighbours classifier
#k option is how many closest neighbours to check class
#stupid sklearn spells neighbours as neighbors 
from sklearn.neighbors import KNeighborsClassifier
#all models are in their own classes called estimator classes.
knn=KNeighborsClassifier(n_neighbors=1)

print(knn.fit(X_train,y_train))

#making predctions
X_new=np.array([[5,2.9,1,0.2]])
print("X_new",X_new.shape)

prediction=knn.predict(X_new)
print("Prediction",prediction)
print("Predicted target name:",iris_dataset['target_names'][prediction])
#predicted to be setosa

#evaluating the model

y_pred=knn.predict(X_test)
print("test set predictions",knn.predict(X_test))

print("Test set score",np.mean(y_pred==y_test))
#mean correctness

#or printed score function
print("test set score",knn.score(X_test,y_test))
