import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC

# Ignore warning to present clean output
warnings.filterwarnings('ignore')

DEBUG = False

if DEBUG:
    TRAINING_DATA = 'trainingDataSubSet.txt'
    TRAINING_TRUTH = 'trainingTruthSubSet.txt'
    TEST_DATA = 'testDataSubSet.txt'
else:
    TRAINING_DATA = 'trainingData.txt'
    TRAINING_TRUTH = 'trainingTruth.txt'
    TEST_DATA = 'testData.txt'
SUBMISSION_FILE = 'TeamEastMeetsWest{}.csv'.format(time.strftime('_%Y%m%d-%H%M%S'))
LOG_FILE = 'output.txt'


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


def replaceMissingValues(X, strategy):
    # Impute missing values, we can choose, mean, median or most frequent
    # Choosing mean as a standard
    imp = Imputer(missing_values='NaN', copy=False, strategy=strategy, axis=0)
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
    with open(LOG_FILE, 'a') as f:
        f.write('Variance Threshold {0}: Keeping {1}, out of {2} features\n'.
                format(threshold, len(kept_features), Xtrain.shape[1]))
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
    # print("Feature ranking:")
    # for f in range(X.shape[1]):
    #    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
    # Plot the feature importances of the forest
    # plt.figure()
    # plt.title("Feature importances")
    # plt.bar(range(Xtrain.shape[1]), importances[indices],
    #        color="r", yerr=std[indices], align="center")
    # plt.xticks(range(Xtrain.shape[1]), indices)
    # plt.xlim([-1, Xtrain.shape[1]])
    # plt.show()
    
    # select features based on importance weights.
    # by default it uses mean importance as the threshold
    selector = SelectFromModel(forest, prefit=True)
        
    # Print out the number of features retained
    kept_features = selector.get_support(indices=True)
    with open(LOG_FILE, 'a') as f:
        f.write('ExtraTreeClassifier: Keeping {0} out of {1} features\n'.format(len(kept_features), Xtrain.shape[1]))
    print('ExtraTreeClassifier: Keeping {0} out of {1} features'.format(len(kept_features), Xtrain.shape[1]))
            
    # Reduce dataset to only include selected features    
    Xtrain = selector.transform(Xtrain)
    Xtest = selector.transform(Xtest)   


def createSubmission(model, Xtest, filename):
    # Create submission
    y_final_prob = model.predict_proba(Xtest)
    y_final_label = model.predict(Xtest)
    
    sample = pd.DataFrame(np.hstack([y_final_prob.round(5),y_final_label.reshape(y_final_prob.shape[0],1)]))
    sample.columns = ['prob1','prob2','prob3','prob4','label']
    sample.label = sample.label.astype(int)
    
    # Submit this file to dropbox
    sample.to_csv(filename, sep='\t', index=False, header=None)
    print('Output for test data written to \'{}\'.'.format(filename))


def getAUCByClass(model, X, Y, classes=[1, 2, 3, 4]):
    
    # Get the predictions
    model_predict = model.predict_proba(X)

    # Binarize the output
    y_bin = label_binarize(Y, classes=classes)
    model.predict_proba(X)
    
    # Calculate AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], model_predict[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    return(roc_auc)  


def main():

    with open(LOG_FILE, 'a') as f:
        f.write('=====\nRun started {}\n\n'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
        f.write('SVC settings:\n')
        f.write('** Scaled both training and test data this time\n')
        f.write('** Running grid search on SVC\n')
        f.write('kernel: poly, random_state=25, cache_size=1000\n\n')

    # Reading files
    read_start = time.time()

    # Read in the data
    X = pd.read_csv(TRAINING_DATA, sep='\t', header=None)
    Y = pd.read_csv(TRAINING_TRUTH, sep='\t', header=None)
    Xtest = pd.read_csv(TEST_DATA, sep="\t", header=None)

    read_end = time.time()

    with open(LOG_FILE, 'a') as f:
        f.write('X.shape: {}\nY.shape: {}\nXtest.shape: {}\n\n'.format(str(X.shape), str(Y.shape), str(Xtest.shape)))

    run_start = time.time()

    # Flatten output labels array
    Y = np.array(Y).ravel()

    # Do some data exploration
    # exploreData(X_scaled)

    # Replace missing values
    replaceMissingValues(X, 'mean')

    # Scale data to normal distribution (gaussian,  mean = 0, variance = 1)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    Xtest_scaled = scaler.transform(Xtest)

    # Reduce feature based on importance
    reduceFeatureswithExtraTrees(Y, X_scaled, Xtest_scaled)

    param_grid = {
        'kernel': ['poly', 'linear', 'sigmoid'],
        'degree': [2, 4, 5],
        'gamma': [.1, .25, .5],
        'C': [.1, .25, .5]
    }

    clf = SVC(probability=True, cache_size=1000)

    gs_start = time.time()
    clf_tuned = GridSearchCV(clf, param_grid=param_grid)
    gs_end = time.time()

    clf_tuned.fit(X_scaled, Y)

    auc_scores = getAUCByClass(clf_tuned, X_scaled, Y, classes=[1, 2, 3, 4])
    with open(LOG_FILE, 'a') as f:
        f.write('best_estimator_:\n')
        f.write(str(clf_tuned.best_estimator_))
        f.write('\nclf_tuned.best_score_ = {}\n'.format(clf_tuned.best_score_))
        f.write('clf_tuned.best_params_ = {}\n'.format(str(clf_tuned.best_params_)))
        f.write('AUC scores by class: {}\n\n'.format(str(auc_scores)))
    print(auc_scores)

    run_end = time.time()

    with open(LOG_FILE, 'a') as f:
        if not DEBUG:
            createSubmission(clf, Xtest_scaled, SUBMISSION_FILE)
            f.write('Submission file created: {}\n'.format(SUBMISSION_FILE))
        f.write('Time to run grid search: {:.3f}s\n'.format(gs_end - gs_start))
        f.write('Time to run analysis: {:.3f}s\n\n'.format(run_end - run_start))
        f.write('Run ended {}\n\n'.format(time.strftime('%Y-%m-%s %H:%M:%S')))

    # Print some timings
    print('Time to load data: {:.3f}s'.format(read_end - read_start))
    print('Time to run analysis: {:.3f}s'.format(run_end - run_start))


if __name__ == '__main__':
    main()
