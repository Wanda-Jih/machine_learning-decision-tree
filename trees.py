import pandas as pd
import numpy as np
import sys
import random

def read_data(input_path):
    # read csv file
    data = pd.read_csv(input_path)
    return data

def decisionTree(train_data, test_data):
    max_depth = 8
    min_children = 50
    
    tree_dict= {}
    feature_list = list(train_data.columns)
    tree_dict = buildTree(train_data, feature_list, max_depth, min_children)
    
    train_acc = evaluate_DT(train_data, tree_dict)
    test_acc = evaluate_DT(test_data, tree_dict)
    
    return train_acc, test_acc
        
def buildTree(train_data, feature_list, max_depth, min_children):
    
    # when run random forest, features only left ['decision']
    if len(feature_list) == 1: 
        return majorityVote(train_data)
    
    choose_feature = ""
    best_gain = -100000
    
    decision_col = train_data['decision']
    decision_count = decision_col.value_counts()
    decision_gini = 1
    for key in decision_count.keys():
        value = decision_count[key]
        decision_gini -= (value/len(decision_col))**2
        
    # select feature with the highest gain
    for feature in feature_list[:-1]:
        learn_col = train_data[feature]
        learn_col_count = learn_col.value_counts()
        feature_gain = decision_gini
        for key in learn_col_count.keys():
            value = learn_col_count[key]
            percentage = value / len(learn_col)
            
            small_gini = 1
            studey_data = train_data[learn_col == key]
            decision_col = studey_data['decision']
            decision_count = decision_col.value_counts()
            for key in decision_count.keys():
                value = decision_count[key]
                small_gini -= (value/len(decision_col))**2 
            
            feature_gain-= percentage * small_gini
        
        if feature_gain > best_gain:
           choose_feature = feature
           best_gain = feature_gain
           
    # put the selected feature to root
    remain_features = feature_list.copy()
    remain_features.remove(choose_feature)

    # create the tree    
    tmp_dict = {}
    tree_dict = {}
    learn_col = train_data[choose_feature]
    learn_col_count = learn_col.value_counts()
       
    tmp_dict['decision'] = majorityVote(train_data)
    for key in learn_col_count.keys():
        value = learn_col_count[key]
        if value <= min_children or max_depth == 1:
            tmp_dict[key] = majorityVote(train_data[train_data[choose_feature] == key])
        else:
            tmp_dict[key] = buildTree(train_data[train_data[choose_feature] == key], remain_features, max_depth-1, min_children)
    
    tree_dict[choose_feature] = tmp_dict
    return tree_dict
        
def majorityVote(train_data):
    max_count = 0
    max_key = 0
    decision_col = train_data['decision']
    decision_count = decision_col.value_counts()
    for key in decision_count.keys():
        value = decision_count[key]
        if value > max_count:
            max_count = value
            max_key = key
    return max_key
 
def evaluate_DT(data, tree_dict):
    
    # check every row. If it matches to the tree's decision, then count += 1
    count = 0
    for i in range(len(data)):
        predict = find_decision_from_tree(data.iloc[i, :], tree_dict)
        if predict == data.iloc[i, -1]:
            count += 1
    
    return count / len(data)

def find_decision_from_tree(row, tree_dict):
    
    for key in tree_dict.keys():
        new_dict = tree_dict[key]
        value = row[key]
        
        if type(new_dict) != dict:
            return new_dict
        if value not in new_dict:
            return new_dict['decision']
                
        decide_dict = new_dict[value]
        # find the last element
        if type(decide_dict) != dict:
            return decide_dict
        else:
            return find_decision_from_tree(row, decide_dict)
   
    
def bagging(train_data, test_data):
    max_depth = 8
    min_children = 50
 
    tree_list = []
    feature_list = list(train_data.columns)
    
    for i in range(30):
        data = train_data.sample(frac = 1, replace = True)
        tmp_dict = buildTree(data, feature_list, max_depth, min_children)
        tree_list.append(tmp_dict)
   
    train_acc = evaluate_BT(train_data, tree_list)
    test_acc = evaluate_BT(test_data, tree_list)
    
    return train_acc, test_acc    
    
def evaluate_BT(data, tree_list):
    count = 0
    for i in range(len(data)):
        decision_dict = {}
        for k in range(len(tree_list)):
            predict = find_decision_from_tree(data.iloc[i, :], tree_list[k])
            if predict in decision_dict:
                decision_dict[predict] += 1
            else:
                decision_dict[predict] = 1
        
        max_count = 0
        max_key = 0   
        for key in decision_dict.keys():
            if decision_dict[key] > max_count:
                max_count = decision_dict[key]
                max_key = key
        
        if max_key == data.iloc[i, -1]:
            count += 1 
    
    return count / len(data)


def randomForests(train_data, test_data):
    max_depth = 8
    min_children = 50
 
    tree_list = []
    
    for i in range(30):
        # collect row
        data = train_data.sample(frac = 0.5, replace = True)
        
        #collect col
        tmp_list = list(data.columns)
        sample_feature = random.sample(tmp_list[:-1], int(np.sqrt(len(tmp_list)-1))) + [tmp_list[-1]]
        
        # build the tree
        data = data[sample_feature]
        tmp_dict = buildTree(data, list(data.columns), max_depth, min_children)
        tree_list.append(tmp_dict)

    train_acc = evaluate_BT(train_data, tree_list)
    test_acc = evaluate_BT(test_data, tree_list)
    
    return train_acc, test_acc   


    
if __name__ == "__main__":
    if(len(sys.argv) != 4):
        trainingSet_path = "trainingSet.csv"
        testSet_path = "testSet.csv"
        modelIdx = 2
        
    else:
        trainingSet_path = sys.argv[1]
        testSet_path = sys.argv[2]
        modelIdx = int(sys.argv[3])

    train_data = read_data(trainingSet_path)
    test_data = read_data(testSet_path)
    if modelIdx == 1:
        trainAcc, testAcc = decisionTree(train_data, test_data)
        print("Training Accuracy DT: %.2f" %trainAcc)
        print("Testing Accuracy DT: %.2f" %testAcc)
        
    elif modelIdx == 2:
        trainAcc, testAcc = bagging(train_data, test_data)
        print("Training Accuracy BT: %.2f" %trainAcc)
        print("Testing Accuracy BT: %.2f" %testAcc)
        
    elif modelIdx == 3:
        trainAcc, testAcc = randomForests(train_data, test_data)
        print("Training Accuracy RT: %.2f" %trainAcc)
        print("Testing Accuracy RT: %.2f" %testAcc)