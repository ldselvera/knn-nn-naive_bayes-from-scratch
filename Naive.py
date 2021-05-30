# -*- coding: utf-8 -*-

#imports
from math import sqrt, pi, exp
from csv import reader
from random import seed,randrange


"""
Helper functions
"""
#calculate probability
def probability(x,avg,standev):
    exponent = exp(-((x-avg)**2 / (2 * standev**2)))
    return (1/(sqrt(2*pi) *standev)) * exponent

#mean
def avg(vals):
    return sum(vals)/float(len(vals))

#standard deviation
def standev(vals):
    mean = avg(vals)
    var = sum([(x-mean)**2 for x in vals]) / float(len(vals)-1)
    return sqrt(var)

"""
Data Handling
"""
def read_csv(file_name):
    data = list()
    with open(file_name, 'r') as file:
        csv = reader(file)
        for row in csv:
            if not row:
                continue
            data.append(row)
    return data

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

def int_from_string_col(data,col):
    class_val =[row[col] for row in data]
    unique_set = set(class_val)
    lookup = dict()
    for i, val in enumerate(unique_set):
        lookup[val] = i
    for row in data:
        row[col] = lookup[row[col]]
    return lookup

def move_class_to_last_col(data,col):
    for row in data:
        temp = row[col]
        del row[col]
        row.append(temp)
    return data


"""
Implementation Functions
"""

"""
We need to calculate the probability of data according to their class so the 
training data needs to be split up by classes. In order to do this we need to 
establish the column that represents the class value for each dataset.   
"""
# this works for datasets with last column representing class value
def split_class(data):
    data_by_class = dict()
    for i in range(len(data)):
        instance = data[i]
        class_val = instance[-1]
        if(class_val not in data_by_class):
            data_by_class[class_val] = list()
        data_by_class[class_val].append(instance)
    return data_by_class

  


"""
We need to find the mean and standard deviation for each column of input.
"""

def data_stats(data):
        stats = [(avg(col),standev(col),len(col)) for col in zip(*data)] 
        del(stats[-1])
        return stats
    
def class_stats(data):
    split = split_class(data)
    class_stats = dict()
    for class_val, row in split.items():
        class_stats[class_val] = data_stats(row)
    return class_stats

"""
Calculate Class Probabilities
"""
def class_get_prob(stats,instance):
    num_rows = sum([stats[label][0][2] for label in stats])
    prob_vals = dict()
    for class_val, class_stats in stats.items():
        prob_vals[class_val] = stats[class_val][0][2]/float(num_rows)
        for i in range(len(class_stats)):
            avg,standev,size = class_stats[i]
            prob_vals[class_val] *= probability(instance[i],avg,standev)
    return prob_vals

def predict(stats,instance):
    prob_vals = class_get_prob(stats,instance)
    top_prob, top_label = -1, None
    for class_val, prob in prob_vals.items():
        if top_label is None or prob > top_prob:
            top_prob = prob
            top_label = class_val
    return top_label

def cross_validation_split(data, n_folds):
	data_split = list()
	copy = list(data)
	fold_size = int(len(data) / n_folds)
	for _ in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(copy))
			fold.append(copy.pop(index))
		data_split.append(fold)
	return data_split
 
def evaluate(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 
def cross_validation(data, algo, n_folds, *args):
	folds = cross_validation_split(data, n_folds)
	accuracy_list = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			copy = list(row)
			test_set.append(copy)
			copy[-1] = None
		predicted = algo(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = evaluate(actual, predicted)
		accuracy_list.append(accuracy)
	return accuracy_list

def naive_bayes(train,test):
    stats = class_stats(train)
    preds = list()
    for row in test:
        result = predict(stats,row)
        preds.append(result)
    return(preds)
    

def run(file_name, target):
    seed(1)
    data = read_csv(file_name)
    data = move_class_to_last_col(data,target)
    for i in range(len(data[0])-1):
        str_column_to_float(data,i)
    int_from_string_col(data,len(data[0])-1)
    n_folds = 10
    accuracies = cross_validation(data, naive_bayes, n_folds)
    print("10-fold Cross-Validation Accuracy Scores")
    for score in accuracies:
        print("%.4f%%" % score)
    print('Mean Accuracy: %.4f%%' % (sum(accuracies)/float(len(accuracies))))
     

     