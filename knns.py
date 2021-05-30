# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from math import sqrt
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

def euclidean_distance(row1, row2):
  #Assure variables are numpy arrays for math operations
  row1, row2 = np.array(row1), np.array(row2)
  #Initial distance
  distance = 0

  for i in range(len(row1)-1):
    #euclidean distance summation (x1-x2)^2
    distance += (row1[i]-row2[i])**2

  #square root of distance accordin to euclidean distance
  return np.sqrt(distance)

def predict(k, train_set, test_instance):
  #list to store distances, neighbors, and classes
  distances = []
  neighbors = []
  classes = {}

  for i in range(len(train_set)):
    #calculate euclidean distances
    dist = euclidean_distance(train_set[i][:-1], test_instance)
    #store euclidean distances
    distances.append((train_set[i], dist))
  
  #sort euclidean distances 
  distances.sort(key=lambda x:x[1])
  
  #find and store k nearest neighbors
  for i in range(k):    
    #get closest distances
    neighbors.append(distances[i][0])
  
  #find label for each instance
  for i in range(len(neighbors)):
    #target datapoint
    response = neighbors[i][-1]

    #if label found increment count
    if response in classes: classes[response] += 1
    else: classes[response] = 1
  
  #Sort classes
  sorted_classes = sorted(classes.items(), key=lambda x: x[1], reverse=True)

  #return prediction
  return sorted_classes[0][0]

def evaluate(y_true, y_pred):
  #number of correct predictions
  n_correct = 0

  #zip used to iterate over two list simultenously 
  for act, pred in zip(y_true, y_pred):
    #add to correct predictions
    if act ==  pred: n_correct += 1

  #return accuracy      
  return n_correct/ len(y_true)

#main function to run KNN
def run(file_name, target, k = 3, ran = 10):
  #read csv without header names, we use column index for processing
  df = pd.read_csv(file_name, header=None)

  #get the column index where values to be predicted is
  target = int(target)
  Y = df[target]

  #If target values are string, we encode using labelencoder
  if isinstance(Y[0], str):
    encoder = LabelEncoder()
    Y = encoder.fit_transform(Y)
    Y = pd.DataFrame(Y)

  #Data preprocessing cleaning
  
  #set the last column to be target values
  df = df.drop(columns=[target])
  df = pd.concat([df, Y], axis=1)

  #drop columns with NANs
  df = df.dropna(axis='columns')
  #drop rows with NANs
  df = df.dropna()

  #get values from dataframe
  X = df.values

  #normalize data by standardize features by removing the mean and scaling to unit variance
  sc = StandardScaler()
  X = sc.fit_transform(X)

  #Seperate test and training data set to 85% and 15% respectively
  # train_set, test_set = train_test_split(train_set, test_size=0.15)

  # #list to store predictions
  # preds = []

  # #create model
  # for row in test_set:
  #   y_test = row[:-1]
  #   pred = predict(k, train_set, y_test)
  #   preds.append(pred)

  # #ground truth labels
  # actual = np.array(test_set)[:, -1]

  # #evaluate accordin to accuracy
  # print("Accuracy:", evaluate(actual, preds))


  kf = KFold(n_splits = 10)
  scores =[]

  print("10-fold Cross-Validation Accuracy Scores")
  
  for train_index, test_index in kf.split(X):

    train_set, test_set = X[train_index], X[test_index]
    
    #list to store predictions
    preds = []

    #create model
    for row in test_set:
      y_test = row[:-1]
      pred = predict(k, train_set, y_test)
      preds.append(pred)

    #ground truth labels
    actual = np.array(test_set)[:, -1]
    score =  evaluate(actual, preds)
    print("%.4f%%" % (score*100))
    scores.append(score)

  print('Mean Accuracy: %.4f%%' % ((sum(scores)/float(len(scores)))*100))

