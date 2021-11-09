# Comparision models
# decision tree, bagging, random forest


### Description

Compare decision tree, bagging, random forest models.
Build these 3 models from scratch without using the library.
Use 10-fold cross validation to evaluate the models.


### Steps for running this repo
1. Execute `preprocess-assg4.py`
    1. Place `dating-full.csv` in the same directory as `preprocess-assg4.py`.
    2. Execute `python .\preprocess-assg4.py`
    3. Output: 
        `trainingSet.csv` and `testSet.csv` will be created in the same direcotry as `preprocess-assg4.py`

2. Execute `trees.py`
* Build decision trees
    1. Execute `python .\trees.py trainingSet.csv testSet.csv 1`
      1. You can choose {1, 2, 3} for the last parameter.
    2. Output: 
        cv_depth.jpg
        
3. Execute `cv_depth.py`
* Build decision trees
    1. Execute `python .\cv_depth.py trainingSet.csv`
    2. Output: 
        cv_depth.jpg
        
4. Execute `cv_frac.py`
* Build decision trees
    1. Execute `python .\cv_frac.py trainingSet.csv`
    2. Output: 
        cv_frac.jpg   
        
5. Execute `cv_numtrees.py`
* Build decision trees
    1. Execute `python .\cv_numtrees.py trainingSet.csv`
    2. Output: 
        cv_numtrees.jpg
