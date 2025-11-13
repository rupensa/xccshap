import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import sys, getopt
import expe_utils as ccshap
import os.path
from SAME.SAME import getExplainer
import warnings
import os
from time import time
import pickle

rng = np.random.RandomState(42)
RANDOM_SEED = 42
MAX_JOBS=16
os.environ["TQDM_DISABLE"] = "1"

def mean_length_path(model, X):
   pathmat = np.array(model.decision_path(X).todense())
   summat = np.count_nonzero(pathmat,axis=1)
   mpl = np.mean(summat)
   stdpl = np.std(summat)
   return mpl, stdpl

def ccshap_full(id_dataset, output_path, model_path, max_depth=100, model_type = XGBClassifier()):
   df_X, df_y, dataset_name, cat_features = ccshap.open_preprocess_uci_dataset(id_dataset)
   print(dataset_name)
   for target_col in df_y.columns:
      class_acc = []
      class_f1 = []
      class_mcc = []
      surr_test_acc = []
      surr_test_f1 = []
      surr_test_mcc = []
      surr_test_avg_pl = []
      surr_test_std_pl = []
      surr_test_nodes = []
      surr_test_depth = []
      surr_test_leaves = []
      nclass = []
      dsname = []
      dstarget = []
      nrows = []
      ncols = []
      nrows_train = []
      ncols_train = []
      nrows_test = []
      ncols_test = []
      total_time = []
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
      class_acc.append(accuracy_score(df_y_test.to_numpy(),y_predicted))
      class_f1.append(f1_score(df_y_test.to_numpy(),y_predicted, average='macro'))
      class_mcc.append(matthews_corrcoef(df_y_test.to_numpy(),y_predicted))
      print('Computing surrogate models...')
      #surr_model=FBTRF(class_model)
      surr_model, tot_time=getExplainer(mode=class_model,originalData=df_X_train,y=df_y_train,oblique_is_explainer=False,agnostic=False,case_six=False)
      depths, y_test_predicted_surr = surr_model.predict(df_X_test.to_numpy(),class_model.feature_names_in_)
      surr_test_acc.append(accuracy_score(y_predicted,y_test_predicted_surr))
      surr_test_f1.append(f1_score(y_predicted,y_test_predicted_surr, average='macro'))
      surr_test_mcc.append(matthews_corrcoef(y_predicted,y_test_predicted_surr))
      mpl = np.mean(depths)
      stdpl = np.std(depths)
      surr_test_nodes.append(surr_model.dt_n_nodes())
      surr_test_leaves.append(surr_model.dt_n_leaves())
      surr_test_depth.append(surr_model.dt_depth())
      surr_test_avg_pl.append(mpl)
      surr_test_std_pl.append(stdpl)
      nclass.append(df_y[target_col].nunique())
      dsname.append(dataset_name)
      dstarget.append(target_col)
      nrows.append(np.shape(df_X_all)[0])
      ncols.append(np.shape(df_X_all)[1])
      nrows_train.append(np.shape(df_X_train)[0])
      ncols_train.append(np.shape(df_X_train)[1])
      nrows_test.append(np.shape(df_X_test)[0])
      ncols_test.append(np.shape(df_X_test)[1])
      total_time.append(tot_time)
      data = {}
      data["dataset"] = dsname
      data["class"] = dstarget
      data["#class"] = nclass
      data["#rows"] = nrows
      data["#cols"] = ncols
      data["#rows_train"] = nrows_train
      data["#cols_train"] = ncols_train
      data["#rows_test"] = nrows_test
      data["#cols_test"] = ncols_test
      data["class_acc"] = class_acc
      data["class_f1"] = class_f1
      data["class_mcc"] = class_mcc
      data["surr_test_acc"] = surr_test_acc
      data["surr_test_f1"] = surr_test_f1
      data["surr_test_mcc"] = surr_test_mcc
      data["surr_test_avg_pl"] = surr_test_avg_pl
      data["surr_test_std_pl"] = surr_test_std_pl
      data["surr_test_nodes"] = surr_test_nodes
      data["surr_test_leaves"] = surr_test_leaves
      data["surr_test_depth"] = surr_test_depth   
      data["total_time"] = total_time
      print('Done.')
      out_table=pd.DataFrame(data)
      output_file_name = "xccshap_full_same_"+dataset_name.replace(" ", "_")+"_"+target_col.replace(" ", "_")+".csv"
      output_file = os.path.join(output_path, output_file_name)
      out_table=pd.DataFrame(data)
      out_table.to_csv(output_file, index=False)
      pkl_file_name = "same_model_"+dataset_name.replace(" ", "_")+"_"+target_col.replace(" ", "_")+".pkl"
      pkl_file = os.path.join(output_path, pkl_file_name)
      with open(pkl_file, 'wb') as fpkl:
         pickle.dump(surr_model, fpkl, pickle.HIGHEST_PROTOCOL)

def main(argv):
   output_path = '.'
   id_dataset = 53
   classifier = 'rf'
   model_path = '.'
   max_depth = 100
   warnings.filterwarnings('ignore') 
   opts, args = getopt.getopt(argv,"hi:o:",["help","out=","id=","max_depth=", "modelpath=","classifier="])
   for opt, arg in opts:
      if opt in ("-h", "--help"):
         print ('ccshap_expe.py --out <outputdir> []')
         sys.exit()
      elif opt in ("-o", "--out"):
         output_path = arg
      elif opt in ("-m", "--modelpath"):
         model_path = arg
      elif opt in ("-m", "--max_depth"):
         max_depth = int(arg)
      elif opt in ("-c", "--classifier"):
         if not(arg in ['rf']):
            sys.exit('Error: classifier model type not supported.')
         classifier = arg
      elif opt in ("-i", "--id"):
         id_dataset = int(arg)
   print("SAME: test using "+classifier+" on: "+str(id_dataset))
   if (classifier=='rf'):
      ccshap_full(id_dataset, output_path, model_path, max_depth=max_depth, model_type=RandomForestClassifier(verbose=0))
if __name__ == "__main__":
   main(sys.argv[1:])