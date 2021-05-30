# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 13:54:44 2020

@author: gerrl
"""
from csv import reader

def select_dataset():
    flag = True
    file_name = ""
    while(flag):
        file_value = (input("Please select the .csv file to be used:\n[1] - Iris.csv\n[2] - wine.csv\n[3] - wdbc.csv\n[4] - Enter custom filename\n"))
        if file_value == '1':
            file_name = "iris.csv"
            flag = False
        elif file_value == '2':
            file_name = "wine.csv"
            flag = False
        elif file_value == '3':
            file_name = "wdbc.csv"
            flag = False
        elif file_value == '4':
            file_name = (input("Enter filename with .csv extention: "))
            flag = False
        else:
            print("\nInvalid Input")
    return file_name
    
def select_classification_value(file_name):
    if file_name == 'iris.csv':
        target = 4
    elif file_name == 'wine.csv':
        target = 0
    elif file_name == 'wdbc.csv':
        target = 1
    else:
        target = input("Please input which column index for the classification value (indexed starting at 0): \n")
    return target

def select_method():
    flag = True
    print("Available classification methods:\n1. KNN\n2. Bayes\n3. Neural Nets")
    while(flag):
        method = input("Which classification method? \n")
        #Choose classification method
        if method == '1':
            clas = 'knn'
            flag = False
        elif method == '2':
            clas = 'bayes'
            flag = False
        elif method == '3':
            clas = 'neural'
            flag = False
        else:
            print("PLEASE CHOOSE A CLASSIFICATION ALGORITHM")
    return clas
