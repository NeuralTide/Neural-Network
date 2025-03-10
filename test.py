import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nn
import sys
import os
sys.path.append('..')
from code_misc.utils import MyUtils

k = 10  #number of classes
d = 784 #number of features, excluding the bias feature

print("Reading Data...")
# READ in data
df_X_train = pd.read_csv('MNIST/x_train.csv', header=None)
df_y_train = pd.read_csv('MNIST/y_train.csv', header=None)
df_X_test = pd.read_csv('MNIST/x_test.csv', header=None)
df_y_test = pd.read_csv('MNIST/y_test.csv', header=None)

print("Saving to numpy arrays...")
# save in numpy arrays
X_train_raw = df_X_train.to_numpy()
y_train_raw = df_y_train.to_numpy()
X_test_raw = df_X_test.to_numpy()
y_test_raw = df_y_test.to_numpy()


print("Getting training set size...")
# get training set size
n_train = X_train_raw.shape[0]
n_test = X_test_raw.shape[0]

print("Normalziing all features...")
# normalize all features to [0,1]
X_all = MyUtils.normalize_0_1(np.concatenate((X_train_raw, X_test_raw), axis=0))
X_train = X_all[:n_train]
X_test = X_all[n_train:]

print("Converting labels...")
# convert each label into a 0-1 vector
y_train = np.zeros((n_train, k))
y_test = np.zeros((n_test, k))
for i in range(n_train):
    y_train[i,int(y_train_raw[i])] = 1.0
for i in range(n_test):
    y_test[i,int(y_test_raw[i])] = 1.0


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
#print(y_test)


print("Building network...")
# build the network
nuts = nn.NeuralNetwork()

nuts.add_layer(d = d)  # input layer - 0

nuts.add_layer(d = 256, act = 'relu')  # hidden layer - 1
nuts.add_layer(d = 256, act = 'relu')  # hiddent layer - 2

nuts.add_layer(d = k, act = 'logis')  # output layer,    multi-class classification, #classes = k



print("Training...")
nuts.fit(X_train, y_train, eta = 0.1, iterations = 50000, SGD = True, mini_batch_size = 20)

print("\n")
print("Miscalc (Train): " + format((nuts.error(X_train, y_train) * 100), '.3f') + "%")
print("Miscalc (Test): " + format((nuts.error(X_test, y_test) * 100), '.3f') + "%")
print("\n")
print("Accuracy (Train): " + format((100 - (nuts.error(X_train, y_train) * 100)), '.3f') + "%")
print("Accuracy (Test): " + format((100 - (nuts.error(X_test, y_test) * 100)), '.3f') + "%")
print("\n")
print("Change in error: " + str(nuts.error_list[0]) + " -> " + str(nuts.error_list[-1]))