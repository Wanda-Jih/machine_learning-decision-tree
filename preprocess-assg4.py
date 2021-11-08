# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# import pandas as pd
import sys
import pandas as pd


def read_data(input_path):
    
    # read csv file
    train_data = pd.read_csv(input_path)
    train_data = train_data.drop(columns=['race', 'race_o', 'field'])
    train_data = train_data.head(6500)

    # assign number to gender column
    train_data = value_assigned(train_data, 0)
    

    # calculate the mean
    train_data = normalization(train_data, 6, 11) 
    train_data = normalization(train_data, 12, 17)   
    
    # discretize all the continuous-valued columns
    divide_list = [1,2]
    for i in range(4, 49):
        divide_list.append(i)
        
    for col in divide_list:
        train_data = divide_dataset(train_data, col)
    
    return train_data

def value_assigned(train_data, col):
    
    for i in range(6500):
        if train_data.at[i, 'gender'] == "male":
            train_data.at[i, 'gender'] = 0
        else:
            train_data.at[i, 'gender'] = 1
            
    return train_data
         
def normalization(train_data, i, j):
    
 
    for row in range(len(train_data)):
        
        sum = 0
        
        # if the sum of the six attributes is not 100
        # need to normalize the data
        for col in range(i, j+1):
            sum += train_data.iloc[row, col]
        
        for col in range(i, j+1):
            value = float(train_data.iloc[row, col]/sum)
            train_data.iloc[row, col] = value
    
    return train_data

def divide_dataset(train_data, col):
    
    max_value = train_data.iloc[:, col].max()
    min_value = train_data.iloc[:, col].min()
    middle_value = min_value + float((max_value - min_value) / 2)

    if (col == 1 or col == 2):
        max_value = 58
        min_value = 18
        middle_value = min_value + float((max_value - min_value) / 2)
        
        
    bins = [min_value, middle_value, max_value]
    labels = [0, 1]
    train_data[train_data.columns[col]] = pd.cut(x = train_data[train_data.columns[col]], bins = bins, labels = labels, include_lowest = True) 

    return train_data

def write_into_csv(train_data, trainingSet_path, testSet_path):
    
    testSet = train_data.sample(frac=0.2, random_state=47)        
    trainSet = train_data.drop(testSet.index)
    
    trainSet.to_csv(trainingSet_path, index = False)
    testSet.to_csv(testSet_path, index = False)
    
if __name__ == "__main__":
    
    if(len(sys.argv) != 3):
        input_path = "dating-full.csv"
        trainingSet_path = "trainingSet.csv"
        testSet_path = "testSet.csv"
        
    else:
        input_path = sys.argv[1]
        trainingSet_path = sys.argv[2]
        testSet_path = sys.argv[3]
    
    train_data = read_data(input_path)
    
    write_into_csv(train_data, trainingSet_path, testSet_path)    