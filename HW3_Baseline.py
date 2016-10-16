#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 1 16:06:15 2016

@author: Hazel John and Cindy Liu
"""
import pandas as pd
import numpy as np
import warnings
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import time

from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC
from fancyimpute import KNN

# Set the parameters for Matplotlib figure size
# for the rest of the Notebook as in section notes
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size

# Ignore warning to present clean output
warnings.filterwarnings('ignore')

# Added ability to debug with smaller datasets
DEBUG = False
DEBUG_FILESIZES = 1000

def initRunID():
    # Generate an ID to identify each run
    global g_runID
    g_runID = '%s'%datetime.now().strftime('%m-%d-%Y_%H%M')
    
def initLogging(name):
    
    # Initialize log confid
    log_fn = './%s_%s.log'%(name,g_runID)
                        
    # create logger 
    global logger
    logger = logging.getLogger('HW3')
    
    # reset handlers so as not to have to exit shell 
    # between two executions
    logger.handlers = []
    
    logger.setLevel(logging.DEBUG)
    
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_fn)
    fh.setLevel(logging.DEBUG)
    
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s',
                                  datefmt='%m/%d/%Y %I:%M:%S %p')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)  
    
def createDataFileSubsets(rowsToCopy, sTrain, sTruth, sTest, sBlind):
    # Name the data subsets to be created
    sTrain_sub = 'trainingDataSubSet.txt'
    sTruth_sub = 'trainingTruthSubSet.txt'
    sTest_sub = 'testDataSubSet.txt'
    sBlind_sub = 'blindDataSubSet.txt'
    
    # Create a list of files to read from and the corresponding files to write to
    data_in = [sTrain, sTruth, sTest, sBlind]
    data_out = [sTrain_sub, sTruth_sub, sTest_sub, sBlind_sub]

    # Read in the files and write out a subset
    for i, data_set in enumerate(data_in):
        with open(data_set, 'r') as inf:
            with open(data_out[i], 'w') as outf:
                for row in range(rowsToCopy):
                    outf.write(inf.readline())
    
    # Return the new filenames
    return (sTrain_sub, sTruth_sub, sTest_sub, sBlind_sub)
    
def readData(sTrain, sTruth, sTest, sBlind):
    # Reading files
    read_start = time.time()
    
    logger.info('Reading files - %s, %s, %s and %s' % (sTrain, sTruth, sTest, sBlind))

    # Read in the data
    X = pd.read_csv(sTrain, sep='\t', header=None)
    Y = pd.read_csv(sTruth, sep='\t', header=None)
    # Flatten output labels array
    Y = np.array(Y).ravel()
    Xtest = pd.read_csv(sTest, sep="\t", header=None)
    Xblind = pd.read_csv(sBlind, sep="\t", header=None)
    # Drop the last column for the Blind data set
    Xblind.drop(Xblind.columns[334], axis=1, inplace=True)
    
    read_end = time.time()
    
    # Print some timings
    logger.info('Time to load data: %0.3fs' % (read_end - read_start))
    
    # Log the size of data
    logger.info('X.shape: %s, Y.shape: %s, Xtest.shape: %s, Xblind.shape: %s' %
        (X.shape, Y.shape, Xtest.shape, Xblind.shape))
    
    return (X, Y, Xtest, Xblind)

def createSubmission(model, Xtest, isBlind):
    #Create submission
    y_final_prob = model.predict_proba(Xtest)
    y_final_label = model.predict(Xtest)
    
    sample = pd.DataFrame(np.hstack([y_final_prob.round(5),y_final_label.reshape(y_final_prob.shape[0],1)]))
    sample.columns = ["prob1","prob2","prob3","prob4","label"]
    sample.label = sample.label.astype(int)
        
    # Create results filename 
    filename = 'TeamEastMeetsWest-%s-%s.csv'%('blind' if isBlind else 'test', g_runID) 

    #Submit this file to dropbox
    sample.to_csv(filename,sep="\t" ,index=False, header=None)
    logger.info('Submission file created: %s' % filename)
            
def exploreData(X):
    # This takes too long with all the rows, so we use a subset
    # This helps us have a quick look at the feature values
    sns.heatmap(X[0:10], xticklabels=20, yticklabels=False)
    plt.show()
    
    # Do some further analysis of the data to see how it is 
    # distributed within the range seen above
    # First check the distribution of column means
    plt.hist(X.mean(axis=0))
    plt.xlabel('mean')
    plt.ylabel('frequency')
    plt.show()

    # Also check distribuyion of column variance
    plt.hist(X.var(axis=0))
    plt.xlabel('variance')
    plt.ylabel('frequency')
    plt.show()
    
def plotFeatureHistograms(X, total_rows, nrows, ncols):
    # Make sure the total_rows we want is not larger than data size
    total_rows = total_rows if total_rows < X.shape[1] else X.shape[1]
    
    # Calculate number of iteration needed to get through all columns
    total_plots = nrows*ncols
    num_iterations = int(total_rows/total_plots + (1 if total_rows%total_plots != 0 else 0))
    start = cur = 0
    
    # Draw a nrows x ncols grid till we run out of rows
    for i in range(num_iterations):
        cur = start*total_plots
        fig = plt.figure()
        for j in range(total_plots):
            if (cur+j >= total_rows):
                break
            ax=fig.add_subplot(nrows, ncols, j+1)
            X[cur+j].hist(ax=ax)
        fig.tight_layout()
        plt.show()
        start += 1
        
def plotFeatureCorrelations(X):
    sns.set(context="paper", font="monospace")

    corrmat = X.corr()
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(14, 10))

    # Draw the heatmap using seaborn
    _ = sns.heatmap(corrmat, vmax=.8, square=True)

def replaceMissingValues(X, strategy):
    # Impute missing values based on the strategy passed in
    # Strategy can be 'mean', 'median' or 'most_frequent'
    imp = Imputer(missing_values='NaN', copy=False, strategy=strategy, axis=0)
    imp.fit_transform(X)   
    logger.info('Missing values replaced using %s' % strategy)
    
def replaceMissingValueswFancyImpute(X):
    # Use 3 nearest rows which have a feature to fill in each row's missing features
    knnImpute = KNN(k=3)
    X_imputed = knnImpute.complete(X)
    return (X_imputed)

def reduceFeaturesbyVariance(Xtrain, Xtest, threshold = 0.22):
    # one way of removing low importance features is to 
    # remove features with low variability
    # Looking at the variance histogram, we can 
    # choose 0.22 as a good cutoff
    selector = VarianceThreshold(threshold = threshold)
    selector.fit(Xtrain)
    
    # Print out the number of features retained
    kept_features = selector.get_support(indices=True)
    logger.info('Variance Threshold %0.2f: Keeping %d, out of %d features' 
                % (threshold, len(kept_features), Xtrain.shape[1]))

    # Reduce dataset to only include selected features
    train_reduced = selector.transform(Xtrain)
    test_reduced = selector.transform(Xtest)
    
    return (train_reduced, test_reduced)
    
def reduceFeatureswithExtraTrees(Y, Xtrain, Xtest):
    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)
    
    forest.fit(Xtrain, Y)
    importances = forest.feature_importances_
    
    # Compute the std. deviations
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    
    # Log top 10 features
    logger.info("Feature ranking:")
    for f in range(10):
        logger.info("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
    # Plot the feature importances of the forest
    plt.title("Feature importances")
    plt.bar(range(Xtrain.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(Xtrain.shape[1]), indices, rotation=90)
    plt.xlim([-1, Xtrain.shape[1]])
    ax = plt.axes()
    # Skip some of the feature labels to reduce crowding
    ax.xaxis.set_major_locator(ticker.MultipleLocator(15))
    ax.set_xlabel('features') 
    ax.set_ylabel('importance')
    plt.show()
    
    # select features based on importance weights.
    # by default it uses mean importance as the threshold
    selector = SelectFromModel(forest, prefit=True)
        
    # Print out the number of features retained
    kept_features = selector.get_support(indices=True)
    logger.info('ExtraTreeClassifier: Keeping %d, out of %d features' %
                (len(kept_features), Xtrain.shape[1]))
            
    # Reduce dataset to only include selected features    
    train_reduced = selector.transform(Xtrain)
    test_reduced = selector.transform(Xtest)
    return (train_reduced, test_reduced)
    
def getAUCByClass(model, X, Y, classes=[1, 2, 3, 4]):
    
    # Get the predictions
    model_predict = model.predict_proba(X)

    # Binarize the output
    y_bin = label_binarize(Y, classes=classes)
    
    #Calculate AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], model_predict[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    return(roc_auc)
    
def getF1ScoreByClass(model, X, Y, classes=[1, 2, 3, 4]):
    
    # Get the predictions
    model_predict = model.predict_proba(X)
    # Binarize the predictions
    model_predict = (model_predict == model_predict.max(axis=1, keepdims=True)).astype(int)

    # Binarize the output
    y_bin = label_binarize(Y, classes=classes)
    
    # Calculate F1 scores
    f1_scores = dict()
    for i in range(4):
        f1_scores[i] = f1_score(y_bin[:, i], model_predict[:, i])
    
    return(f1_scores)
    
def runRandomForestwithGridSearch(Y, Xtrain, Xtest, isBlind):
    
    # Note time to run this setup 
    run_start = time.time()
    
    # Reduce feature based on importance
    (Xtrain, Xtest) = reduceFeatureswithExtraTrees(Y, Xtrain, Xtest)
    
    # Specify the parameters to tune
    param_grid = {'estimator__n_estimators':[15, 20], 
                  'estimator__max_depth':[8, 10], 
                  'estimator__min_samples_split':[50, 100],
                  'estimator__min_samples_leaf':[10, 20],
                  'estimator__max_features': ['sqrt', 0.25]}

    gs_start = time.time()
    model_to_set = OneVsRestClassifier(RandomForestClassifier(random_state=25, oob_score = True), -1)

    model_tuning = GridSearchCV(model_to_set, 
                            param_grid = param_grid, 
                            scoring='f1_weighted',
                            iid=False,
                            n_jobs=-1)
    # Fit the model
    model_tuning.fit(Xtrain, Y)
        
    gs_end = time.time()
    logger.info('Time to run grid search (RandomForest): %0.3fs'% (gs_end - gs_start))

    logger.info('Best score = %d' % model_tuning.best_score_)
    logger.info('Best params = %s' % model_tuning.best_params_)
    logger.info('AUC per class = %s' % 
                getAUCByClass(model_tuning, Xtrain, Y, classes=[1, 2, 3, 4]))
    logger.info('F1 Score per class = %s' % 
                getF1ScoreByClass(model_tuning, Xtrain, Y, classes=[1, 2, 3, 4]))
                
    # Create submission file    
    createSubmission(model_tuning, Xtest, isBlind)
    
    # Note the end time
    run_end = time.time()
    logger.info('Time to run analysis(RandomForest): %0.3fs'% (run_end - run_start))

    
def runSVMwithGridSearch(Y, Xtrain, Xtest, isBlind):
    
    # Note time to run this setup 
    run_start = time.time()
       
    # Normalize data since accuracy of SVM can severely degrade if it isn't
    # Scale data to normal distribution (gaussian,  mean = 0, variance = 1)
    scaler = StandardScaler().fit(Xtrain)
    X_scaled = scaler.transform(Xtrain)
    Xtest_scaled = scaler.transform(Xtest)
    
    # Reduce feature based on importance
    (X_scaled, Xtest_scaled) = reduceFeatureswithExtraTrees(Y, X_scaled, Xtest_scaled)
    
    gs_start = time.time()
    # Use default values for C and other parameters
    param_grid = [{'kernel': ['rbf'], 'gamma': [0.01, 0.1]},
                  #{'kernel': ['linear']},
                  {'kernel': ['poly'], 'degree': [2, 3]}]
    
    clf = SVC(probability=True)
    
    clf_tuned = GridSearchCV(clf, param_grid=param_grid, cv=3,
                       scoring='f1_weighted', n_jobs=-1)
    
    # Fit the model
    clf_tuned.fit(X_scaled, Y)
    
    gs_end = time.time()
    logger.info('Time to run grid search(SVC): %0.3fs'% (gs_end - gs_start))

    logger.info('Best score = %d' % clf_tuned.best_score_)
    logger.info('Best params = %s' % clf_tuned.best_params_)
    logger.info('AUC per class = %s' % 
                getAUCByClass(clf_tuned, X_scaled, Y, classes=[1, 2, 3, 4]))
    logger.info('F1 Score per class = %s' % 
                getF1ScoreByClass(clf_tuned, X_scaled, Y, classes=[1, 2, 3, 4]))
    
    # Predict for the test data and create submission
    createSubmission(clf_tuned, Xtest_scaled, isBlind)
    
    run_end = time.time()
    logger.info('Time to run analysis(SVC): %0.3fs'% (run_end - run_start))

def runDecisionTreewithAdaboost(Y, Xtrain, Xtest, isBlind):
    
    # Note time to run this setup 
    run_start = time.time()
    
    # Reduce feature based on importance
    (Xtrain, Xtest) = reduceFeatureswithExtraTrees(Y, Xtrain, Xtest)
    
    model_start = time.time()
       
    # Specify parameters for GridSearch
    param_grid = {'base_estimator__criterion' : ["gini", "entropy"], 
                  'base_estimator__max_depth':[8, 10], 
                  'base_estimator__max_features':['sqrt', 0.25],
                  'n_estimators': [25, 30],
                  'learning_rate': [0.8, 1.0]}

    dtc = DecisionTreeClassifier(random_state = 11, max_features = "auto", 
                                 class_weight = "balanced")
    abc = AdaBoostClassifier(base_estimator = dtc, algorithm="SAMME.R")
    
    # run grid search
    abc_tuned = GridSearchCV(abc, param_grid=param_grid, scoring='f1_weighted')

    # Fit the model
    abc_tuned.fit(Xtrain, Y)
    
    model_end = time.time()
    logger.info('Time to run Gridsearch with AdaBoost(DecisionTree): %0.3fs'% (model_end - model_start))
        
    logger.info('Model params = %s' % abc_tuned.get_params())
    logger.info('AUC per class = %s' % 
                getAUCByClass(abc_tuned, Xtrain, Y, classes=[1, 2, 3, 4]))
    logger.info('F1 Score per class = %s' % 
                getF1ScoreByClass(abc_tuned, Xtrain, Y, classes=[1, 2, 3, 4]))
                
    # Create submission file    
    createSubmission(abc_tuned, Xtest, isBlind)
    
    # Note the end time
    run_end = time.time()
    logger.info('Time to run analysis(AdaBoost): %0.3fs'% (run_end - run_start))

def main():
    
    # Specify the data files we need
    sTrain = 'trainingData.txt'
    sTruth = 'trainingTruth.txt'
    sTest = 'testData.txt'
    sBlind = 'blindData.txt'

    # Create a smaller set of files to use for debugging, and update 
    # file names to point to the new set
    if (DEBUG):
        (sTrain, sTruth, sTest, sBlind) = createDataFileSubsets(DEBUG_FILESIZES, sTrain, sTruth, sTest, sBlind)
    
    (X, Y, Xtest, Xblind) = readData(sTrain, sTruth, sTest, sBlind)
    
    # Do some data exploration
    exploreData(X)
    
    # Look at the histogram of the first 48 rows in 4x4 grids
    plotFeatureHistograms(X, 48, 4, 4)
    
    # Look at the correlations
    plotFeatureCorrelations(X)
    
    # Impute missing values, we can choose, mean, median or most frequent
    # Choosing mean as we didn't notice skewness in the attribute distribution
    replaceMissingValues(X, strategy = 'mean')
    
    # Fancyimpute requires installation of other packages
    # We can use this or the simple imputation shown above
    #X = replaceMissingValueswFancyImpute(X)
    
    # Run randomforest classifier with gridsearch
    #runRandomForestwithGridSearch(Y, X, Xtest, False)
    
    # Run SVM classifier with gridsearch
    runSVMwithGridSearch(Y, X, Xblind, True)
    
    # Run DecisionTree classifier with AdaBoost
    #runDecisionTreewithAdaboost(Y, X, Xtest, False)
    
if __name__=='__main__':
    
    # generate a runid
    initRunID()

    # initialize logging
    initLogging('HW3_run')
    
    logger.info("Starting run ...")
    
    # Call the main function
    main()
