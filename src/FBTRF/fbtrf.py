import numpy as np
import datetime
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from FBTRF.ConjunctionSet import ConjunctionSet
from FBTRF.Branch import Branch
import os
from FBTRF.NewModelBuilder import *
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import cohen_kappa_score



class FBTRF():
    def __init__(self,rf):
        self.number_of_branches_threshold = 3000
        self.rf=rf

    def fit(self,df_X,df_Y):
        cs = ConjunctionSet(df_X.columns, df_X.to_numpy(),df_X.to_numpy(),df_Y.to_numpy(), self.rf, df_X.dtypes,self.number_of_branches_threshold)
        #Train the new model
        self.branches_df = cs.get_conjunction_set_df().round(decimals=5)
        for i in range(2):
            self.branches_df[self.rf.classes_[i]] = [probas[i] for probas in self.branches_df['probas']]
        df_dict = {}
        for col in self.branches_df.columns:
            df_dict[col] = self.branches_df[col].values
        self.model = Node([True]*len(self.branches_df))
        self.model.split(df_dict)

    def predict(self,X):
        probas,depths=[],[]
        for inst in X.to_numpy():
            prob,depth=self.model.predict_probas_and_depth(inst,self.branches_df)
            probas.append(prob)
            depths.append(depth)
        predictions=[self.rf.classes_[i] for i in np.array([np.argmax(prob) for prob in probas])]
        return predictions, depths
    
    def dt_n_nodes(self):
        return self.model.number_of_children()

    def dt_depth(self):
        return self.model.count_depth()
    
    def dt_n_leaves(self):
        return self.model.number_of_leaves()