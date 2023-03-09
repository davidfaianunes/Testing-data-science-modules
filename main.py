import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

# Dataset:
# Dua, D. and Graff, C.(2019). UCI Machine Learning Repository[http://archive.ics.uci.edu/ml]. Irvine, CA:
# University of California, School of Information and Computer Science.
# Donated by: P- Savicky Institute of Computer Science, AS of CR Czech Republic savick '@' cs.cas.cz

cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("magic04.data", names=cols) #changing cols_names to cols
df.head() # .head() gives back the first 5 items/lines in dataframe

df["class"] = (df["class"]=="g").astype(int)

for label in cols[:-1]:
  plt.hist(df[df["class"]==1][label], color = "blue", label = "gamma", alpha =0.7, density = True)
  plt.hist(df[df["class"]==0][label], color = "red", label = "hadron", alpha =0.7, density = True)
  plt.title(label)
  plt.ylabel("Probability")
  plt.xlabel(label)
  plt.legend()
  plt.show()
  
# Training, validation, testing datasets

def scale_dataset(dataframe, oversample = False): # assume that the label is the last collumn
  X = dataframe[dataframe.columns[:-1]].values  # all collumns minus the "label"
  Y = dataframe[dataframe.columns[-1]].values  # labels

  scaler = StandardScaler() # to fit and transform
                            # AKA fix the fact that the data is in different orders ofmagnitudes
  X = scaler.fit_transform(X)

  if oversample:
    ros = RandomOverSampler()
    X, Y = ros.fit_resample(X, Y)

  # create whole data as 2D-numpy-array
  data = np.hstack((X, np.reshape(Y, (-1,1))))
  #NOTE 1: np.hstack() stacks horizontally
  #NOTE 2: numpy is picky with dimensions, and being Y a 1D-array, we have to make it a 2D-array
        # (-1,1) is (num_rows, num_cols) where -1 tells the computer to compute len(Y)

  return data, X, Y
  
#print(len(train[train["class"]==1]))     7428  
#print(len(train[train["class"]==0]))     3984

# this servers to examplify how the training dataset is not homogenous

train, x_train, y_train = scale_dataset(train, oversample = True) #training dataset
valid, x_valid, y_valid = scale_dataset(valid, oversample = False) #validation dataset
test, x_test, y_test = scale_dataset(test, oversample = False)     #testing dataset

# validation and testing datasets dont need to be oversampled, they just check the validity of our model
# should just be a random set of data

#print(sum(y_train == 1))
#print(sum(y_train == 0))
#print()
#print(sum(y_test == 1))
#print(sum(y_test == 0))
#print()
#print(sum(y_valid == 1))
#print(sum(y_valid == 0))

#kNN

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train, y_train)

y_pred = knn_model.predict(x_test)

print(classification_report(y_test, y_pred))

#  Naive Bayes

from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(x_train, y_train)

y_pred = nb_model.predict(x_test)

print(classification_report(y_test, y_pred))

# Logistic Regression

from sklearn.linear_model import LogisticRegression

lg_model = LogisticRegression()
lg_model = lg_model.fit(x_train, y_train)

y_pred = lg_model.predict(x_test)

print(classification_report(y_test, y_pred))

# SVM

from sklearn.svm import SVC

svm_model = SVC()
svm_model = svm_model.fit(x_train, y_train)

y_pred = svm_model.predict(x_test)

print(classification_report(y_test, y_pred))

# Neural Net

#import tensorflow as tf
#
#nn_model = tf.keras.Sequential([
#    tf.keras.Layers.Dense(32, activation ="relu", input_shape=(10,)),
#    tf.keras.Layers.Dense(32, activation ="relu"),
#    tf.keras.layers.Dense(1, activation="sigmoid") #output layer
#])
#
#nn_model.compile(optimizer=tf.keras.optimizer.adam)

#incomplete




