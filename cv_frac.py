import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def read_data(input_path):
    
    data = pd.read_csv(input_path)
    data = data.sample(random_state = 18, frac = 1)
    # data = data.sample(random_state = 32, frac = 0.5)
    return data    

def build_folder(train_data, folder_num):
    
    division = len(train_data) / folder_num
    folder_list = []
    for i in range(folder_num):
        new_fold = train_data.iloc[int(i*division): int((i+1)*division), :]
        folder_list.append(new_fold)
    
    return folder_list
   
def build_trainingset(data_list, index, t_frac):
    temp_row = []
    for i, row in enumerate(data_list):
        if i == index:
            continue
        temp_row.append(row)
    training_set = pd.concat(temp_row)
    training_set = training_set.sample(random_state = 32, frac = t_frac)
    return training_set


def decisionTree(train_data, test_data, max_depth, min_children):
    
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
  
 
def bagging(train_data, test_data, max_depth, min_children):
 
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


def randomForests(train_data, test_data, max_depth, min_children):
 
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
   
    
if __name__ == '__main__':
   
    if len(sys.argv) == 1:
        input_path = "trainingSet.csv"
    else:
        input_path = sys.argv[1]
        
    train_data = read_data(input_path)
    
    t_frac = [0.05, 0.075, 0.1, 0.15, 0.2]
    folder_num = 10
    min_children = 50
    max_depth = 8
    trees_num = 30
    
    data_list = build_folder(train_data, folder_num)


    DT_data = []
    BT_data = []
    RF_data = []    

    
    for t in t_frac:
        DT_record = []
        BT_record = []
        RF_record = []
        
        for i in range(folder_num):
            test_set = data_list[i]
            training_set = build_trainingset(data_list, i, t)
            
            # decision tree
            trainAcc, testAcc = decisionTree(training_set, test_set, max_depth, min_children)
            DT_record.append(testAcc)
            
            # bagging
            trainAcc, testAcc = bagging(training_set, test_set, max_depth, min_children)
            BT_record.append(testAcc)        
            
            # random forest
            trainAcc, testAcc = randomForests(training_set, test_set, max_depth, min_children)
            RF_record.append(testAcc)   
         
        result = stats.ttest_rel(RF_record, BT_record)
        print(result)
        

        #Here append the mean_accu and err
        DT_mean=np.mean(DT_record)
        DT_Var=np.var(DT_record)
        DT_sterr=np.sqrt(DT_Var)/np.sqrt(folder_num)
        DT_data.append([max_depth,DT_mean,DT_sterr])

        BT_mean = np.mean(BT_record)
        BT_Var = np.var(BT_record)
        BT_sterr = np.sqrt(BT_Var) / np.sqrt(folder_num)
        BT_data.append([max_depth, BT_mean, BT_sterr])

        RF_mean = np.mean(RF_record)
        RF_Var = np.var(RF_record)
        RF_sterr = np.sqrt(RF_Var) / np.sqrt(folder_num)
        RF_data.append([max_depth, RF_mean, RF_sterr])
        
        #Save tmp result
        np.savetxt("tmp_result/DT_record_frac.txt",np.array(DT_data))
        np.savetxt("tmp_result/bt_record_frac.txt", np.array(BT_data))
        np.savetxt("tmp_result/rf_record_fracc.txt", np.array(RF_data))
    



    # Plot the result
    DT_data = np.array(DT_data)
    BT_data = np.array(BT_data)
    RF_data = np.array(RF_data)

    plt.errorbar(t_frac, DT_data[:, 1], yerr=DT_data[:, 2], label='Decision Tree')
    plt.errorbar(t_frac, BT_data[:, 1], yerr=BT_data[:, 2], label='Bagging Tree')
    plt.errorbar(t_frac, RF_data[:, 1], yerr=RF_data[:, 2], label='Random Forest')
    plt.xlabel('Training fraction')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('cv_frac.jpg')
        