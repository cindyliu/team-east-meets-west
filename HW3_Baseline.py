import pandas as pd
import numpy as np
import time
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.grid_search import GridSearchCV


# Ignore warning to present clean output
warnings.filterwarnings('ignore')

def createSubmission(model, filename):
    #Create submission
    Xtest = pd.read_csv("data/testData.txt",sep="\t",header=None)
    y_final_prob = model.predict_proba(Xtest)
    y_final_label = model.predict(Xtest)
    
    sample = pd.DataFrame(np.hstack([y_final_prob.round(5),y_final_label.reshape(y_final_prob.shape[0],1)]))
    sample.columns = ["prob1","prob2","prob3","prob4","label"]
    sample.label = sample.label.astype(int)
    #Submit this file to dropbox
    sample.to_csv(filename,sep="\t" ,index=False, header=None)
    print("Output for test data written to :", filename)
    
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
    
#Reading files
read_start = time.time()
X = pd.read_csv("data/trainingData.txt",sep='\t',header=None)
Y = pd.read_csv("data/trainingTruth.txt",sep='\t',header=None)
read_end = run_start = time.time()
Y = np.array(Y).ravel()

# Check if missing values
cols_wmissing = X.columns[pd.isnull(X).any()].tolist()
print(len(cols_wmissing))

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
    'estimator__n_estimators': [25, 50],
    'estimator__criterion': ['gini', 'entropy'],
    'estimator__max_features': ['log2', 'sqrt', 0.25],
    'estimator__max_depth': [10, 20, None]
    #'estimator__oob_score': [True, False]
}

model_to_set = OneVsRestClassifier(RandomForestClassifier(random_state=25), -1)

model_tuning = GridSearchCV(model_to_set, param_grid=param_grid, 
                             scoring='f1_weighted')
# Fit the model
model_tuning.fit(X, Y)
print(model_tuning.best_score_)
print(model_tuning.best_params_)
best_params = model_tuning.best_params_
print(getAUCByClass(model_tuning, X, Y, classes=[1, 2, 3, 4]))

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
createSubmission(model_tuning, "TeamEastMeetsWest.csv")

print('Time to load data: {:.3f}s'.format(read_end - read_start))
print('Time to run analysis: {:.3f}s'.format(run_end - run_start))
