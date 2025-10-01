import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import sys, getopt
from xgboost import XGBClassifier
from XCCShap.xccshap import XCCShap
from XCCShap.xccshapsurrogate import XCCShapSurrogate
import expe_utils as ccshap
import os.path
import warnings
from scipy.stats import ttest_ind, ranksums, mannwhitneyu
import os
from time import time
import pickle

rng = np.random.RandomState(42)
RANDOM_SEED = 42
MAX_JOBS=16
os.environ["TQDM_DISABLE"] = "1"


def mean_length_path(model, X, n_features):
   idx = np.where(model.dt_model.tree_.__getstate__()['nodes']['feature']>=n_features)[0]
   pathmat = np.array(model.decision_path(X).todense())
   summat = np.count_nonzero(pathmat,axis=1)-np.count_nonzero(pathmat[:,idx],axis=1)
   mpl = np.mean(summat)
   stdpl = np.std(summat)
   return mpl, stdpl

def mean_length_path_baseline(model, X):
   pathmat = np.array(model.decision_path(X).todense())
   summat = np.count_nonzero(pathmat,axis=1)
   mpl = np.mean(summat)
   stdpl = np.std(summat)
   return mpl, stdpl

def mean_length_path_fbt(model, X):
   dplist = model.get_decision_paths(X)
   dparray = np.zeros(len(dplist))
   for i in range(len(dplist)):
      dparray[i] = len(dplist[i])-1
   mpl = np.mean(dparray)
   stdpl = np.std(dparray)
   return mpl, stdpl

def ccshap_full(id_dataset, output_path, model_path, model_type = RandomForestClassifier(), explainer='kernel', decorr=False):
   df_X, df_y, dataset_name, cat_features = ccshap.open_preprocess_uci_dataset(id_dataset)
   print(dataset_name)
   for target_col in df_y.columns:
      match_avg = []
      match_std = []
      mismatch_avg = []
      mismatch_std = []
      pvalueg = []
      pvaluel = []
      print('Using target: ',target_col)
      df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df_X, df_y[target_col], test_size=0.3, random_state=rng)
      df_X_all = df_X.to_numpy()
      print('Loading model...')
      model_file_name = "dataset_"+str(id_dataset)+"_"+target_col.replace(" ", "_")
      model_file = os.path.join(model_path, model_file_name)
      class_model = ccshap.load_model(model_type, model_file)
      if class_model is None:
         continue
      y_predicted = class_model.predict(df_X_test)
      dsname = []
      dstarget = []
      print('Loading surrogate models...')
      pkl_file_name = "xccshap_model_"+dataset_name.replace(" ", "_")+"_"+target_col.replace(" ", "_")+".pkl"
      pkl_file = os.path.join(output_path, pkl_file_name)
      with open(pkl_file, 'rb') as f:
         surr_model = pickle.load(f)
      y_test_predicted_surr, sim_values =surr_model.predict(df_X_test)
      sim_match_ind = np.where(y_predicted==y_test_predicted_surr)[0].tolist()
      sim_unmatch_ind = np.where(y_predicted!=y_test_predicted_surr)[0].tolist()
      sim_match = [sim_values[i] for i in sim_match_ind]
      sim_mismatch = [sim_values[i] for i in sim_unmatch_ind]
      pvalueg.append(ttest_ind(a=sim_match, b=sim_mismatch, alternative='greater', equal_var=False).pvalue.item())
      pvaluel.append(ttest_ind(a=sim_match, b=sim_mismatch, alternative='less', equal_var=False).pvalue.item())
      match_avg.append(np.mean(sim_match))
      match_std.append(np.std(sim_match))
      mismatch_avg.append(np.mean(sim_mismatch))
      mismatch_std.append(np.std(sim_mismatch))
      dsname.append(dataset_name)
      dstarget.append(target_col)
      data = {}
      data["dataset"] = dsname
      data["class"] = dstarget
      data["match_avg"] = match_avg
      data["match_std"] = match_std
      data["mismatch_avg"] = mismatch_avg
      data["mismatch_std"] = mismatch_std
      data["pvalueg"] = pvalueg
      data["pvaluel"] = pvaluel
      print('Done.')
      out_table=pd.DataFrame(data)
      output_file_name = "xccshap_simvalues_xccshap_"+dataset_name.replace(" ", "_")+"_"+target_col.replace(" ", "_")+".csv"
      output_file = os.path.join(output_path, output_file_name)
      out_table=pd.DataFrame(data)
      out_table.to_csv(output_file, index=False)


def main(argv):
   output_path = '.'
   id_dataset = 53
   classifier = 'xgb'
   model_path = '.'
   warnings.filterwarnings('ignore') 
   opts, args = getopt.getopt(argv,"hi:o:",["help","out=","id=","modelpath=","classifier="])
   for opt, arg in opts:
      if opt in ("-h", "--help"):
         print ('ccshap_expe.py --out <outputdir> []')
         sys.exit()
      elif opt in ("-o", "--out"):
         output_path = arg
      elif opt in ("-m", "--modelpath"):
         model_path = arg
      elif opt in ("-c", "--classifier"):
         if not(arg in ['xgb','rf']):
            sys.exit('Error: classifier model type not supported.')
         classifier = arg
      elif opt in ("-i", "--id"):
         id_dataset = int(arg)
   print("CCSHAP: test using "+classifier+" on: "+str(id_dataset))
   if (classifier=='xgb'):
      ccshap_full(id_dataset, output_path, model_path, model_type=XGBClassifier(verbosity=0,  tree_method='hist', device='cuda'), explainer='tree')
   elif (classifier=='rf'):
      ccshap_full(id_dataset, output_path, model_path, model_type=RandomForestClassifier(verbose=0), explainer='tree')
if __name__ == "__main__":
   main(sys.argv[1:])