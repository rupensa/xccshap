import numpy as np
from SAME.MergeOTrees import mergeDecisionTrees,rec_buildTree,buildTreefromOblique
from SAME.SuperTree import *
import SAME.MergeTrees
import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import RandomizedSearchCV

import time 


def getExplainer(mode,originalData=None,y=None,oblique_is_explainer=True,agnostic=True,case_six=False):
    if type(mode)==DecisionTreeClassifier:
        return mode,0        
    if type(mode)==ensemble.RandomForestClassifier:
        forest = mode
        if type(forest.estimators_[0])==DecisionTreeClassifier:
            #print("bb is an orthogonal forest with",len(forest.estimators_))
            #roots = np.array([rec_buildTree(t,FI_used) for t,FI_used in zip(forest.estimators_,forest.estimators_features_)])
            #roots = np.array([rec_buildTree(t,FI_used) for t,FI_used in zip(forest.estimators_,forest.feature_names_in_)])
            roots = np.array([rec_buildTree(t,forest.feature_names_in_) for t in forest.estimators_])
            startTime = time.time()
            superT = SAME.MergeTrees.mergeDecisionTrees(roots, num_classes = np.unique(y).shape[0])
            return superT,time.time()- startTime
        elif case_six==False:
            print("not orthogonal forest")
            if agnostic:
                return getAgnosticExplainerWithX(mode,originalData)
            else:
                n_class = len(set(y))
                roots = np.array([buildTreefromOblique(t._root,n_class,FI_used) for t,FI_used in zip(forest.estimators_,forest.estimators_features_)])
                startTime = time.time()
                superT = mergeDecisionTrees(roots,
                                            num_classes = np.unique(y).shape[0],
                                            OriginalPoints = originalData,
                                           maximum_depth=10)
                return superT,time.time()- startTime
        else: #Case six
            print("A FOREST")
            roots=[]
            for r in forest.estimators_:
                y_tr = r.predict(originalData)
                f_j = getForest(md=[4,5,6],n_est=2,x_tr=originalData,y_tr=y_tr)
                for t,FI_used in zip(f_j.estimators_,f_j.estimators_features_):
                    prune_duplicate_leaves(t)
                    roots.append(rec_buildTree(t,FI_used))
            print("CASE SIX",len(forest.estimators),"estimators"," merging",len(roots),"trees")
            startTime = time.time()
            superT = MergeTrees.mergeDecisionTrees(roots, num_classes = np.unique(y).shape[0])
            return superT,time.time()- startTime
    else:
        return getAgnosticExplainerWithX(mode,originalData)
    
def getForest(x_tr,y_tr,md=[4,5],n_est=2):
    param_dict =  {
    'base_estimator__max_depth': md,
    'base_estimator__class_weight': [None, 'balanced'],
    }
    Itermediate_forest = ensemble.BaggingClassifier(base_estimator=DecisionTreeClassifier())
    random_search =  RandomizedSearchCV(Itermediate_forest,param_dict,iid=True,cv=3,refit=bestRFRefit)
    result = random_search.fit(x_tr,y_tr)
    forest = result.best_estimator_
    return forest

def getAgnosticExplainerWithX(black_box,originalData):
    X_trainO = originalData[np.random.choice(originalData.shape[0],int(originalData.shape[0]/5))]
    yBB = black_box.predict(X_trainO)
    RF = ensemble.BaggingClassifier(base_estimator=DecisionTreeClassifier())
    param_dict =  {
    'n_estimators': [20],
    'base_estimator__min_samples_split': [2, 0.002, 0.01, 0.05, 0.1, 0.2],
    'base_estimator__min_samples_leaf': [1, 0.001, 0.01, 0.05, 0.1, 0.2],
    'base_estimator__max_depth': [4, 5,6,7],
    'base_estimator__class_weight': [None, 'balanced'],
    'base_estimator__random_state': [0],
    }
    random_search =  RandomizedSearchCV(RF,param_dict,iid=True,cv=3,refit=bestRFRefit)
    result = random_search.fit(X_trainO,yBB)
    forest = result.best_estimator_
    print('I am an EXPLAINER')
    for w in ["param_n_estimators","param_base_estimator__max_depth"]:
        print(w,random_search.cv_results_[w][random_search.best_index_])
    for r in forest.estimators_:
        prune_duplicate_leaves(r)
    roots = np.array([rec_buildTree(t,FI_used) for t,FI_used in zip(forest.estimators_,forest.estimators_features_)])
    startTime = time.time()
    superT = MergeTrees.mergeDecisionTrees(roots, num_classes = np.unique(yBB).shape[0])
    return superT,time.time() - startTime


def bestTreeRefit(cv_results):
    res = pd.DataFrame(cv_results)
    ooo = res.sort_values(["rank_test_score","param_max_depth"],ascending=[True,True]).index[0]
    return ooo
def bestRFRefit (cv_results):
    res = pd.DataFrame(cv_results)
    return res.sort_values(["rank_test_score","param_n_estimators","param_base_estimator__max_depth"],ascending=[True,True,True]).index[0]
import time

def trepan(mode,originalData,md=0):    
    param_dict =  {
    'max_depth': [2,3,4,6,7,8,9,10,11,12,14,15,17,20],
    }
    x_tr = originalData
    y_tr = mode.predict(originalData)
    if md == 0:
        random_search =  RandomizedSearchCV(DecisionTreeClassifier(),param_dict,iid=True,cv=3,refit=bestTreeRefit)
        result = random_search.fit(x_tr,y_tr)
        clf = result.best_estimator_
        elapsed = result.refit_time_
    else:
        clf = DecisionTreeClassifier(max_depth=md)
        start = time.time()
        clf.fit(x_tr,y_tr)
        elapsed = start-time.time()
    return clf,elapsed

