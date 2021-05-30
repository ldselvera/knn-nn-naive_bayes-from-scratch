# -*- coding: utf-8 -*-
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

def run_test():
  """ Iris Dataset"""
  from sklearn.datasets import load_iris

  iris = load_iris()
  X = iris.data
  y = iris.target

  kfold = 10
  print("Iris Results")
  print("\nKNN")
  neigh = KNeighborsClassifier(n_neighbors=3)
  scores = cross_val_score(neigh, X, y, cv=kfold)
  print("10-fold Cross-Validation Accuracy Scores: ", (scores*100))
  print('Mean Accuracy: %.4f%%' % ((sum(scores)/kfold)*100))

  print("\nNaive Bayes")
  clf = GaussianNB()
  print("10-fold Cross-Validation Accuracy Scores: ", (scores*100))
  print('Mean Accuracy: %.4f%%' % ((sum(scores)/kfold)*100))

  print("\nNeural Networks")
  clf = MLPClassifier()
  scores = cross_val_score(clf, X, y, cv=kfold)
  print("10-fold Cross-Validation Accuracy Scores: ", (scores*100))
  print('Mean Accuracy: %.4f%%' % ((sum(scores)/kfold)*100))

  """ Wine Dataset"""
  from sklearn.datasets import load_wine

  wine = load_wine()
  X = wine.data
  y = wine.target
  X_train, X_test, y_train, y_test = train_test_split(X, y)

  print("\nWine Results")
  print("KNN")
  neigh = KNeighborsClassifier(n_neighbors=3)
  scores = cross_val_score(neigh, X, y, cv=kfold)
  print("10-fold Cross-Validation Accuracy Scores: ", (scores*100))
  print('Mean Accuracy: %.4f%%' % ((sum(scores)/kfold)*100))

  print("\nNaive Bayes")
  clf = GaussianNB()
  scores = cross_val_score(clf, X, y, cv=kfold)
  print("10-fold Cross-Validation Accuracy Scores: ", (scores*100))
  print('Mean Accuracy: %.4f%%' % ((sum(scores)/kfold)*100))

  print("\nNeural Networks")
  clf = MLPClassifier()
  scores = cross_val_score(clf, X, y, cv=kfold)
  print("10-fold Cross-Validation Accuracy Scores: ", (scores*100))
  print('Mean Accuracy: %.4f%%' % ((sum(scores)/kfold)*100))

  """\n\nBreast Cancer Dataset"""
  from sklearn.datasets import load_breast_cancer

  cancer = load_breast_cancer()
  X = cancer.data
  y = cancer.target
  X_train, X_test, y_train, y_test = train_test_split(X, y)

  print("\nKNN")
  neigh = KNeighborsClassifier(n_neighbors=3)
  scores = cross_val_score(neigh, X, y, cv=kfold)
  print("10-fold Cross-Validation Accuracy Scores: ", (scores*100))
  print('Mean Accuracy: %.4f%%' % ((sum(scores)/kfold)*100))

  print("\nNaive Bayes")
  clf = GaussianNB()
  scores = cross_val_score(clf, X, y, cv=kfold)
  print("10-fold Cross-Validation Accuracy Scores: ", (scores*100))
  print('Mean Accuracy: %.4f%%' % ((sum(scores)/kfold)*100))

  print("\nNeural Networks")
  clf = MLPClassifier()
  scores = cross_val_score(clf, X, y, cv=kfold)
  print("10-fold Cross-Validation Accuracy Scores: ", (scores*100))
  print('Mean Accuracy: %.4f%%' % ((sum(scores)/kfold)*100))