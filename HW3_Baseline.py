import pandas as pd
import numpy as np
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import time

from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import Imputer

# Ignore warning to present clean output
warnings.filterwarnings('ignore')

<<<<<<< HEAD

OUTPUT_FILE = 'TeamEastMeetsWest2.csv'
LOG_FILE = 'output.txt'


def createSubmission(model):
    #Create submission
    Xtest = pd.read_csv('testData.txt',sep='\t',header=None)
=======
def exploreData(X):
    # Do initial analysis of the data
    plt.hist(X.var(axis=0))
    plt.xlabel('variance')
    plt.ylabel('frequency')
    plt.show()
    plt.hist(X.mean(axis=0))
    plt.xlabel('mean')
    plt.ylabel('frequency')
    plt.show()
    
    # This takes too long with all the rows, so we use a subset
    # We see a similar range of values in all columns
    sns.heatmap(X[0:20], xticklabels=20, yticklabels=False)

def replaceMissingValues(X):
    # Impute missing values, we can choose, mean, median or most frequent
    # Choosing mean as a standard
    imp = Imputer(missing_values='NaN', copy=False, strategy='mean', axis=0)
    imp.fit_transform(X)   

def reduceFeaturesbyVariance(Xtrain, Xtest, threshold = 0.22):
    # one way of removing low importance features is to 
    # remove features with low variability
    # We know we have a number below 0.22, so we could
    # remove them
    selector = VarianceThreshold(threshold = threshold)
    selector.fit(Xtrain)
    
    # Print out the number of features retained
    kept_features = selector.get_support(indices=True)
    print('Variance Threshold {0}: Keeping {1}, out of {2} features'.
            format(threshold, len(kept_features), Xtrain.shape[1]))

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
    
    # Print the feature ranking
    #print("Feature ranking:")
    #for f in range(X.shape[1]):
    #    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(Xtrain.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(Xtrain.shape[1]), indices)
    plt.xlim([-1, Xtrain.shape[1]])
    plt.show()
    
    # select features based on importance weights.
    # by default it uses mean importance as the threshold
    selector = SelectFromModel(forest, prefit=True)
        
    # Print out the number of features retained
    kept_features = selector.get_support(indices=True)
    print('ExtraTreeClassifier: Keeping {0}, out of {1} features'.
            format(len(kept_features), Xtrain.shape[1]))
            
    # Reduce dataset to only include selected features    
    Xtrain = selector.transform(Xtrain)
    Xtest = selector.transform(Xtest)   

def createSubmission(model, Xtest, filename):
    #Create submission
>>>>>>> 2e0d3c7b4729b3ac6cd8fbb9db24eec0fc09f931
    y_final_prob = model.predict_proba(Xtest)
    y_final_label = model.predict(Xtest)
    
    sample = pd.DataFrame(np.hstack([y_final_prob.round(5),y_final_label.reshape(y_final_prob.shape[0],1)]))
    sample.columns = ['prob1','prob2','prob3','prob4','label']
    sample.label = sample.label.astype(int)
    
    #Submit this file to dropbox
    sample.to_csv(OUTPUT_FILE, sep='\t', index=False, header=None)
    print('Output for test data written to \'{}\'.'.format(OUTPUT_FILE))
    
def getAUCByClass(model, X, Y, classes=[1, 2, 3, 4]):
    
    # Get the predictions
    model_predict = model.predict_proba(X)

    # Binarize the output
    y_bin = label_binarize(Y, classes=classes)
    model.predict_proba(X)
    
    #Calculate AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], model_predict[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    return(roc_auc)  

<<<<<<< HEAD

try:
    #Reading files
    read_start = time.time()
    X = pd.read_csv('trainingData.txt',sep='\t',header=None)
    Y = pd.read_csv('trainingTruth.txt',sep='\t',header=None)
    read_end = run_start = time.time()
    Y = np.array(Y).ravel()

    with open(LOG_FILE, 'a') as log_fd:
        log_fd.write('=====\nRun data for {}\n'.format(time.strftime('%Y-%m-%d %H:%M')))
        # Check if missing values
        cols_wmissing = X.columns[pd.isnull(X).any()].tolist()
        log_fd.write('Number of columns with missing values: {}\n'.format(len(cols_wmissing)))

        # Replace NaNs for column with column mean (or we could do median)
        X = X.apply(lambda x: x.fillna(x.median()),axis=0)

        #clf = svm.SVC(kernel='linear', C=1)
        #scores = cross_val_score(clf, X, Y, cv=5)
        ##Simple K-Fold cross validation. 10 folds.
        #cv = cross_validation.KFold(len(X), n_folds=10, random_state=41, shuffle=True)
        #results = []
        ## "Error_function" can be replaced by the error function of your analysis
        #for traincv, testcv in cv:
        #    svm.fit(X[traincv], Y[traincv])
        #    svm.predict(X[traincv], Y[traincv])
        #    c1 = svm.score(X[traincv], Y[traincv])
        #    #c2 = svm.score(X[testcv], Y[testcv])
        #    #results.append(svm.score(X[testcv], Y[testcv]))
        #print(scores)

        # If I specify larger number of estimators, it picks the larger ones
        # So, choosing a smaller number deliberately
        param_grid = {
            'estimator__n_estimators': [5, 10, 20],
            'estimator__criterion': ['gini', 'entropy'],
            'estimator__max_features': ['log2', 'sqrt', 0.25, 5],
            'estimator__max_depth': [10, 15, 20],
            'estimator__oob_score': [True, False]
        }

        model_to_set = OneVsRestClassifier(RandomForestClassifier(random_state=25), -1)

        gs_start = time.time()
        model_tuning = GridSearchCV(model_to_set, param_grid=param_grid,
                                     scoring='f1_weighted')
        # Fit the model
        model_tuning.fit(X, Y)
        gs_end = time.time()
        log_fd.write('best_estimator_ :\n')
        log_fd.write(str(model_tuning.best_estimator_))
        log_fd.write('\nmodel_tuning.best_score_ = {}\n'.format(model_tuning.best_score_))
        log_fd.write('model_tuning.best_params_ = ')
        log_fd.write(str(model_tuning.best_params_))
        # best_params = model_tuning.best_params_
        log_fd.write('\nAUC by class: ')
        auc_start = time.time()
        log_fd.write(str(getAUCByClass(model_tuning, X, Y, classes=[1, 2, 3, 4])))

        ## Use cross validation to create the next model
        #model2 = RandomForestClassifier(n_estimators=100)
        ##Simple K-Fold cross validation. 10 folds.
        #cv = cross_validation.KFold(len(X), n_folds=10, indices=False)
        #results = []
        ## "Error_function" can be replaced by the error function of your analysis
        #for traincv, testcv in cv:
        #        probas = model2.fit(X[traincv], Y[traincv]).predict_proba(X[testcv])
        #        results.append( Error_function )
        #
        #model1 = OneVsRestClassifier(RandomForestClassifier(n_estimators = 50, criterion = 'entropy',
        #                            max_depth = 20, max_features= 0.25, random_state=25), -1)
        #model1.fit(X,Y)
        #print(getAUCByClass(model1, X, Y, classes=[1, 2, 3, 4]))
        run_end = time.time()
        createSubmission(model_tuning)

        log_fd.write('\n\nTime to load data: {:.3f}s\n'.format(read_end - read_start))
        log_fd.write('Time to run grid search: {:.3f}s\n'.format(gs_end - gs_start))
        log_fd.write('Time to run AUC: {:.3f}s\n'.format(run_end - auc_start))
        log_fd.write('Total time to run analysis: {:.3f}s\n\n'.format(run_end - run_start))

    print('Log written to \'{}\'.'.format(LOG_FILE))
except SystemExit:
    log_fd.write('\n\nSystem exited at {}.'.format(time.strftime('%Y-%m-%d %H:%M')))
    print('\nSystem exited at {}.'.format(time.strftime('%Y-%m-%d %H:%M')))
    raise
=======
def main():
    # Reading files
    read_start = time.time()
    
    # Read in the data
    X = pd.read_csv("data/trainingData.txt",sep='\t',header=None)
    Y = pd.read_csv("data/trainingTruth.txt",sep='\t',header=None)
    Xtest = pd.read_csv("data/testData.txt",sep="\t",header=None)
    
    read_end = run_start = time.time()
    
    # Flatten output labels array
    Y = np.array(Y).ravel()
    
    # Do some data exploration
    exploreData(X)
    
    # Replace missing values
    replaceMissingValues(X)
    
    # Reduce feature based on importance
    reduceFeatureswithExtraTrees(Y, X, Xtest)
    
    
#    clf = svm.SVC(kernel='linear', C=1)
#    scores = cross_val_score(clf, X, Y, cv=5)
#    #Simple K-Fold cross validation. 10 folds.
#    cv = cross_validation.KFold(len(X), n_folds=10, random_state=41, shuffle=True)
#    results = []
#    # "Error_function" can be replaced by the error function of your analysis
#    for traincv, testcv in cv:
#        svm.fit(X[traincv], Y[traincv])
#        svm.predict(X[traincv], Y[traincv])
#        c1 = svm.score(X[traincv], Y[traincv])
#        #c2 = svm.score(X[testcv], Y[testcv])
#        #results.append(svm.score(X[testcv], Y[testcv]))
#    print(scores)
#    
#     If I specify larger number of estimators, it picks the larger ones
#     So, choosing a smaller number deliberately
#    param_grid = { 
#        'estimator__n_estimators': [25, 50],
#        'estimator__criterion': ['gini', 'entropy'],
#        'estimator__max_features': ['log2', 'sqrt', 0.25],
#        'estimator__max_depth': [10, 20, None],
#        'estimator__oob_score': [True, False]
#    }
#    
#    model_to_set = OneVsRestClassifier(RandomForestClassifier(random_state=25), -1)
#    
#    model_tuning = GridSearchCV(model_to_set, param_grid=param_grid, 
#                                 scoring='f1_weighted')
#    # Fit the model
#    model_tuning.fit(X, Y)
#    print(model_tuning.best_score_)
#    print(model_tuning.best_params_)
#    best_params = model_tuning.best_params_
#    print(getAUCByClass(model_tuning, X, Y, classes=[1, 2, 3, 4]))
#    
#       
#    param_grid = {'estimator__n_estimators':[20,30,40,50], 
#                  'estimator__max_depth':[3,6,8,12,24,32], 
#                  'estimator__min_samples_split':[2,4,6],
#                  'estimator__min_samples_leaf':[1,2,4] }
#    model_to_set = OneVsRestClassifier(RandomForestClassifier(random_state=25), -1)
#    model_tuning = GridSearchCV(model_to_set, 
#                            param_grid = param_grid, 
#                            scoring='f1_weighted',
#                            n_jobs=4, iid=False, cv=5)
#    # Fit the model
#    model_tuning.fit(X, Y)
#    
#    print(model_tuning.best_score_)
#    print(model_tuning.best_params_)
#    print(getAUCByClass(model_tuning, X, Y, classes=[1, 2, 3, 4]))
    
      
    model1 = OneVsRestClassifier(RandomForestClassifier(n_estimators = 50, criterion = 'entropy', 
                                max_depth = 20, max_features= 0.25, random_state=25, oob_score = True), -1)
    model1.fit(X, Y)
    print(getAUCByClass(model1, X, Y, classes=[1, 2, 3, 4]))   
    
    run_end = time.time()
    file_id = '%s'%datetime.now().strftime('%m-%d-%Y_%H%M')
    
    # Take a note of strategy
    strategy_fn = 'Strategy-%s.txt'%file_id
    file = open(strategy_fn, "w")
    
    file.write("Missing values = mean, feature selection = ET")
    file.write("RF = n_estimators = 50, criterion = entropy, max_depth = 20, max_features= 0.25, random_state=25, oob_score = True")
    file.close()
    
    # Create results filename 
    result_fn = 'TeamEastMeetsWest-%s.csv'%file_id  
    
    createSubmission(model1, Xtest, result_fn)
    
    # Print some timings
    print('Time to load data: {:.3f}s'.format(read_end - read_start))
    print('Time to run analysis: {:.3f}s'.format(run_end - run_start))

if __name__=='__main__':
    main()
>>>>>>> 2e0d3c7b4729b3ac6cd8fbb9db24eec0fc09f931
