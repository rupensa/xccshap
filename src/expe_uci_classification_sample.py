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
import os
from time import time

rng = np.random.RandomState(42)
RANDOM_SEED = 42
MAX_JOBS=16
os.environ["TQDM_DISABLE"] = "1"


def _vars(a, axis=None):
    a_squared = a.copy()
    a_squared.data **= 2
    return a_squared.mean(axis) - np.square(a.mean(axis))

def _stds(a, axis=None):
    return np.sqrt(_vars(a, axis))

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
      samplesize = []
      nclust = []
      nclass = []
      dsname = []
      dstarget = []
      nrows = []
      ncols = []
      nrows_train = []
      ncols_train = []
      nrows_test = []
      ncols_test = []
      taux = []
      tauy = []
      class_acc = []
      class_f1 = []
      class_mcc = []
      surr_test_acc = []
      surr_test_f1 = []
      surr_test_mcc = []
      surr_acc_test_acc = []
      surr_acc_test_f1 = []
      surr_acc_test_mcc = []
      surr_test_avg_pl = []
      surr_test_std_pl = []
      surr_test_nodes = []
      surr_test_depth = []
      surr_test_leaves = []
      shap_time = []
      exp_time = []
      surr_time = []
      total_time = []
      print('Using target: ',target_col)
      df_X_train_orig, df_X_test, df_y_train_orig, df_y_test = train_test_split(df_X, df_y[target_col], test_size=0.3, random_state=rng)
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
      datasize = min(np.shape(df_X_train_orig)[0]//2, 100)
      sample_size = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
      K = [int(len(df_X_train_orig)*r) for r in sample_size]
      for k in K:
         for run in range(10):
            print(f'sample size: {k}')
            print('Explaining model...')
            sample = None
            df_X_train = df_X_train_orig.sample(n=k, random_state=rng.randint(0,200))
            df_y_train = df_y_train_orig[df_X_train.index]
            xccshap_model = XCCShap(explainer=explainer, model=class_model, data=sample)
            try:
               shapmat, row_labels, col_labels, exp_time_val, shap_time_val = xccshap_model.explain(df_X_train)
               tau_x = xccshap_model.tau_x_
               tau_y = xccshap_model.tau_y_
               print('Computing surrogate models...')
               surr_model=XCCShapSurrogate(xccshap_model)
               start_time = time()
               surr_model.fit(df_X_train,df_y_train)
            except:
               print(f'failed with sample size: {k}')
               continue
            end_time = time()
            surr_time_val = end_time-start_time
            samplesize.append(k)
            y_test_predicted_surr=surr_model.predict(df_X_test)
            surr_test_acc.append(accuracy_score(y_predicted,y_test_predicted_surr))
            surr_test_f1.append(f1_score(y_predicted,y_test_predicted_surr, average='macro'))
            surr_test_mcc.append(matthews_corrcoef(y_predicted,y_test_predicted_surr))
            taux.append(tau_x)
            tauy.append(tau_y)
            surr_test_nodes.append(surr_model.dt_n_nodes())
            surr_test_depth.append(surr_model.dt_depth())
            surr_test_leaves.append(surr_model.dt_n_leaves())
            shap_time.append(shap_time_val)
            exp_time.append(exp_time_val)
            surr_time.append(surr_time_val)
            total_time.append(shap_time_val+exp_time_val+surr_time_val)
            mpl, stdpl = mean_length_path(surr_model, df_X_test, np.shape(df_X_test)[1])
            surr_test_avg_pl.append(mpl)
            surr_test_std_pl.append(stdpl)
            nclust.append(len(np.unique(row_labels)))
            nclass.append(df_y[target_col].nunique())
            dsname.append(dataset_name)
            dstarget.append(target_col)
            nrows.append(np.shape(df_X_all)[0])
            ncols.append(np.shape(df_X_all)[1])
            nrows_train.append(np.shape(df_X_train_orig)[0])
            ncols_train.append(np.shape(df_X_train_orig)[1])
            nrows_test.append(np.shape(df_X_test)[0])
            ncols_test.append(np.shape(df_X_test)[1])
      data = {}
      data['sample'] = samplesize
      data["dataset"] = dsname
      data["class"] = dstarget
      data["#class"] = nclass
      data["#clust"] = nclust
      data["#rows"] = nrows
      data["#cols"] = ncols
      data["#rows_train"] = nrows_train
      data["#cols_train"] = ncols_train
      data["#rows_test"] = nrows_test
      data["#cols_test"] = ncols_test
      data["taux"] = taux
      data["tauy"] = tauy
      data["surr_test_acc"] = surr_test_acc
      data["surr_test_f1"] = surr_test_f1
      data["surr_test_mcc"] = surr_test_mcc
      data["surr_test_avg_pl"] = surr_test_avg_pl
      data["surr_test_std_pl"] = surr_test_std_pl
      data["surr_test_nodes"] = surr_test_nodes
      data["surr_test_leaves"] = surr_test_leaves
      data["surr_test_depth"] = surr_test_depth
      data["shap_time"] = shap_time
      data["exp_time"] = exp_time
      data["surr_time"] = surr_time
      data["total_time"] = total_time
      print('Done.')
      out_table=pd.DataFrame(data)
      output_file_name = "xccshap_sample_xccshap_"+dataset_name.replace(" ", "_")+"_"+target_col.replace(" ", "_")+".csv"
      output_file = os.path.join(output_path, output_file_name)
      out_table=pd.DataFrame(data)
      out_table.to_csv(output_file, index=False)

def main(argv):
   output_path = ''
   id_dataset = 53
   classifier = 'xgb'
   model_path = '.'
   warnings.filterwarnings('ignore') 
   opts, args = getopt.getopt(argv,"hi:o:",["help","out=","id=","modelpath=","params=","classifier="])
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