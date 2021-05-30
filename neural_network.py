# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

def neural_network(X_train, Y_train, epochs=10, neurons=[], lr=0.15):
    # hidden_layers = len(neurons) - 1
    #Set weights to random weights
    weights = initialize_weights(neurons)

    #Iterate through data according to number of epochs
    for epoch in range(1, epochs+1):
        #train data and get weights resulting from training
        weights = train(X_train, Y_train, lr, weights)

        #Every 10 epochs output accuracy from training
        # if(epoch % 10 == 0):
        #   print("Epoch ", epoch)
        #   #Print training accuracy
        #   print("Training Accuracy: ", accuracy(X_train, Y_train, weights)) 
    
    #return weights from all epochs training
    return weights

def initialize_weights(neurons):
    layers, weights = len(neurons), []
    
    #initialize weights with random values between [-1, 1] including bias
    for i in range(1, layers):
        w = [[np.random.uniform(-1, 1) for k in range(neurons[i-1] + 1)]
            for j in range(neurons[i])]
        weights.append(np.matrix(w))
    
    #return weights with random values between [-1, 1]
    return weights

def forward_propagation(x, weights, layers):
    activations, layer_input = [x], x

    #iterate over all layers
    for j in range(layers):
        #apply sigmoid function
        activation = sigmoid(np.dot(layer_input, weights[j].T))
        #store activation result
        activations.append(activation)
        #output of previous layer becomes input of next layer
        layer_input = np.append(1, activation) 
    
    #return activations results
    return activations

def back_propagation(y, activations, weights, layers, lr):
    #we start back_propagation from last layer, the last activation results
    outputFinal = activations[-1]

    #error at output
    error = np.matrix(y - outputFinal) 
    
    #iterate over all layers starting at last one
    for j in range(layers, 0, -1):
        currActivation = activations[j]
        
        #augment previous activation
        if(j > 1): prevActivation = np.append(1, activations[j-1])
        #first hidden layer, previous activation is input without bias
        else: prevActivation = activations[0]
        
        # backpropagated error of current layer multiplied by activation of previous layer
        delta = np.multiply(error, sigmoid_derivative(currActivation))

        #update weights between current layer and previous layer
        weights[j-1] += lr * np.multiply(delta.T, prevActivation)

        #remove bias from weights
        w = np.delete(weights[j-1], [0], axis=1) 
        #calculate error current layer
        error = np.dot(delta, w) 
    
    return weights

def train(X, Y, lr, weights):
    layers = len(weights)
    for i in range(len(X)):
        x, y = X[i], Y[i]
        x = np.matrix(np.append(1, x)) 
        
        activations = forward_propagation(x, weights, layers)
        weights = back_propagation(y, activations, weights, layers, lr)

    return weights

#Activation function to pass the dot product of each layer through to get final output
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return np.multiply(x, 1-x)

def predict(item, weights):
    layers = len(weights)
    item = np.append(1, item)
    
    #forward propagation
    activations = forward_propagation(item, weights, layers)
    
    outputFinal = activations[-1].A1
    index = max_activation(outputFinal)

    #initialize prediction vector to zeros
    y = [0 for i in range(len(outputFinal))]
    #intialize predicted class to 1
    y[index] = 1  

    #return prediction vector
    return y 


def max_activation(output):
    #Find max activation in output
    m, index = output[0], 0

    for i in range(1, len(output)):
        if(output[i] > m): m, index = output[i], i
    
    return index

def accuracy(X, Y, weights):
    #iterate through network to find accuracy
    correct = 0

    for i in range(len(X)):
        x, y = X[i], list(Y[i])
        pred = predict(x, weights)

        #correctly predicted
        if(y == pred): correct += 1

    #return accuracy
    return correct / len(X)
    
def run(file_name, target, lr = 0.15, epochs = 100, no_neurons = [5, 10]):
  
  #read csv file
  df = pd.read_csv(file_name, header=None)

  #get column number that contains target values, values to be predicted
  target = int(target)

  #Data clearning drop NaNs
  df = df.dropna(axis='columns')
  df = df.dropna()

  #features
  X = df.loc[:, df.columns != target].values
  
  #values to predict, target
  Y = df[target]

  #normalize training data
  sc = StandardScaler()
  X = sc.fit_transform(X)

  #one hot encode classes to creat numerical values, vectors
  one_hot_encoder = OneHotEncoder(sparse=False)
  Y = one_hot_encoder.fit_transform(np.array(Y).reshape(-1, 1))   

  #number of features
  f = len(X[0]) 
  #number of output classes
  o = len(Y[0]) 

  #initialize each layers with only 1
  layers = [1]*(len(no_neurons)+2)

  #first layer number of neurons will be the number of features
  layers[0] = f
  #last layer number of neurons will be the number of output classes
  layers[-1] = o

  #set number of neurons for hidden layers
  for i in range(len(no_neurons)): layers[i+1] = no_neurons[i]
  kf = KFold(n_splits=10)
  scores =[]
  print("10-fold Cross-Validation Accuracy Scores")
  for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
  #store model, essentially the trained weights
    model = neural_network(X_train, Y_train, epochs=epochs, neurons=layers, lr=lr)
    score =  accuracy(X_test, Y_test, model)
    print("%.4f%%" % (score*100))
    scores.append(score)

  print('Mean Accuracy: %.4f%%' % ((sum(scores)/float(len(scores)))*100))