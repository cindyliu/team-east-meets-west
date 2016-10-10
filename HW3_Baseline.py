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

# Ignore warning to present clean output
warnings.filterwarnings('ignore')

# Added ability to debug with smaller datasets
DEBUG = False
if DEBUG:
    TRAINING_DATA = 'trainingDataSubSet.txt'
    TRAINING_TRUTH = 'trainingTruthSubSet.txt'
    TEST_DATA = 'testDataSubSet.txt'
else:
    TRAINING_DATA = 'trainingData.txt'
    TRAINING_TRUTH = 'trainingTruth.txt'
    TEST_DATA = 'testData.txt'

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
    ch.setLevel(logging.ERROR)
    
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s',
                                  datefmt='%m/%d/%Y %I:%M:%S %p')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)  
    
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

def replaceMissingValues(X):
    # Impute missing values, we can choose, mean, median or most frequent
    # Choosing mean as a standard
    strategy = 'mean'
    imp = Imputer(missing_values='NaN', copy=False, strategy=strategy, axis=0)
    imp.fit_transform(X)   
    logger.info('Missing values replaced using %s' % strategy)

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
    Xtrain = selector.transform(Xtrain)
    Xtest = selector.transform(Xtest)
    
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
    Xtrain = selector.transform(Xtrain)
    Xtest = selector.transform(Xtest)   
    
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
    
def createSubmission(model, Xtest, filename):
    #Create submission
    y_final_prob = model.predict_proba(Xtest)
    y_final_label = model.predict(Xtest)
    
    sample = pd.DataFrame(np.hstack([y_final_prob.round(5),y_final_label.reshape(y_final_prob.shape[0],1)]))
    sample.columns = ["prob1","prob2","prob3","prob4","label"]
    sample.label = sample.label.astype(int)
    
    #Submit this file to dropbox
    sample.to_csv(filename,sep="\t" ,index=False, header=None)
    logger.info('Submission file created: %s' % filename)
    
def runRandomForestwithGridSearch(Y, Xtrain, Xtest):
    
    # Note time to run this setup 
    run_start = time.time()
    
    # Reduce feature based on importance
    reduceFeatureswithExtraTrees(Y, Xtrain, Xtest)
    
    # Specify the parameters to tune
    param_grid = {'estimator__n_estimators':[20, 30], 
                  'estimator__max_depth':[10, 20], 
                  'estimator__min_samples_split':[4, 6],
                  'estimator__min_samples_leaf':[2, 4],
                  'estimator__max_features': ['sqrt', 0.25]}
    
    model_to_set = OneVsRestClassifier(RandomForestClassifier(random_state=25, oob_score = True), -1)

    gs_start = time.time()
    model_tuning = GridSearchCV(model_to_set, 
                            param_grid = param_grid, 
                            scoring='f1_weighted',
                            iid=False,
                            n_jobs=-1)
    gs_end = time.time()
    logger.info('Time to run grid search (RandomForest): %0.3fs'% (gs_end - gs_start))

    # Fit the model
    model_tuning.fit(Xtrain, Y)
        
    logger.info('Best score = %d' % model_tuning.best_score_)
    logger.info('Best params = %s' % model_tuning.best_params_)
    logger.info('AUC per class = %s' % 
                getAUCByClass(model_tuning, Xtrain, Y, classes=[1, 2, 3, 4]))
    logger.info('F1 Score per class = %s' % 
                getF1ScoreByClass(model_tuning, Xtrain, Y, classes=[1, 2, 3, 4]))
                
    # Create results filename 
    result_fn = 'TeamEastMeetsWest-%s.csv'%g_runID  
    
    createSubmission(model_tuning, Xtest, result_fn)
    
    # Note the end time
    run_end = time.time()
    logger.info('Time to run analysis(RandomForest): %0.3fs'% (run_end - run_start))

    
def runSVMwithGridSearch(Y, Xtrain, Xtest):
    
    # Note time to run this setup 
    run_start = time.time()
       
    # Normalize data since accuracy of SVM can severely degrade if it isn't
    # Scale data to normal distribution (gaussian,  mean = 0, variance = 1)
    scaler = StandardScaler().fit(Xtrain)
    X_scaled = scaler.transform(Xtrain)
    Xtest_scaled = scaler.transform(Xtest)
    
    # Reduce feature based on importance
    reduceFeatureswithExtraTrees(Y, X_scaled, Xtest_scaled)
    
    param_grid = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3]},
                  {'kernel': ['linear']},
                  {'kernel': ['poly'], 'degree': [2, 3]}]
    
    clf = SVC(probability=True)
    
    gs_start = time.time()
    clf_tuned = GridSearchCV(clf, param_grid=param_grid, cv=3,
                       scoring='f1_weighted', n_jobs=-1)
    gs_end = time.time()
    
    logger.info('Time to run grid search(SVC): %0.3fs'% (gs_end - gs_start))

    # Fit the model
    clf_tuned.fit(X_scaled, Y)

    logger.info('Best score = %d' % clf_tuned.best_score_)
    logger.info('Best params = %s' % clf_tuned.best_params_)
    logger.info('AUC per class = %s' % 
                getAUCByClass(clf_tuned, X_scaled, Y, classes=[1, 2, 3, 4]))
    logger.info('F1 Score per class = %s' % 
                getF1ScoreByClass(clf_tuned, Xtrain, Y, classes=[1, 2, 3, 4]))
                
    # Create results filename 
    result_fn = 'TeamEastMeetsWest-%s.csv'%g_runID  
    
    # Predict for the test data and create submission
    createSubmission(clf_tuned, Xtest_scaled, result_fn)
    
    run_end = time.time()
    logger.info('Time to run analysis(SVC): %0.3fs'% (run_end - run_start))

def runDecisionTreewithAdaboost(Y, Xtrain, Xtest):
    
    # Note time to run this setup 
    run_start = time.time()
    
    # Reduce feature based on importance
    reduceFeatureswithExtraTrees(Y, Xtrain, Xtest)
    
    model_start = time.time()
    model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                               n_estimators=50,
                               learning_rate=1.5,
                               algorithm="SAMME")
    model_end = time.time()
    logger.info('Time to run AdaBoost(DecisionTree): %0.3fs'% (model_end - model_start))

    # Fit the model
    model.fit(Xtrain, Y)
        
    logger.info('Model params = %s' % model.get_params())
    logger.info('AUC per class = %s' % 
                getAUCByClass(model, Xtrain, Y, classes=[1, 2, 3, 4]))
    logger.info('F1 Score per class = %s' % 
                getF1ScoreByClass(model, Xtrain, Y, classes=[1, 2, 3, 4]))
                
    # Create results filename 
    result_fn = 'TeamEastMeetsWest-%s.csv'%g_runID  
    
    createSubmission(model, Xtest, result_fn)
    
    # Note the end time
    run_end = time.time()
    logger.info('Time to run analysis(AdaBoost): %0.3fs'% (run_end - run_start))

def main():
    
    # Reading files
    read_start = time.time()
    
    # Read in the data
    X = pd.read_csv(TRAINING_DATA, sep='\t', header=None)
    Y = pd.read_csv(TRAINING_TRUTH, sep='\t', header=None)
    Xtest = pd.read_csv(TEST_DATA, sep="\t", header=None)
    
    read_end = time.time()
    
    # Print some timings
    logger.info('Time to load data: %0.3fs' % (read_end - read_start))

    # Log the size of data
    logger.info('X.shape: %s, Y.shape: %s, Xtest.shape: %s' %
        (X.shape, Y.shape, Xtest.shape))
        
    # Flatten output labels array
    Y = np.array(Y).ravel()
    
    # Do some data exploration
    exploreData(X)
    
    # Replace missing values
    replaceMissingValues(X)
    
    # Run randomforest classifier with gridsearch
    #runRandomForestwithGridSearch(Y, X, Xtest)
    
    # Run SVM classifier with gridsearch
    #runSVMwithGridSearch(Y, X, Xtest)
    
    # Run DecisionTree classifier with AdaBoost
    runDecisionTreewithAdaboost(Y, X, Xtest)
    
if __name__=='__main__':
    
    # generate a runid
    initRunID()

    # initialize logging
    initLogging('HW3_run')
    
    logger.info("Starting run ...")
    
    # Call the main function
    main()
