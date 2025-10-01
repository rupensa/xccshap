import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sys, getopt
from xgboost import XGBClassifier
import expe_utils as ccshap
import os.path
import warnings
import os
import json

rng = np.random.RandomState(42)
RANDOM_SEED = 42
MAX_JOBS=16
os.environ["TQDM_DISABLE"] = "1"


def ccshap_full(id_dataset, output_path, classifier, model_type = RandomForestClassifier(), encode_labels=True):
   df_X, df_y, dataset_name, cat_features = ccshap.open_preprocess_uci_dataset(id_dataset, encode_labels=encode_labels)
   print(dataset_name)
   for target_col in df_y.columns:
      for rnd_seed in range(10):
         rnd = np.random.RandomState(rnd_seed)
         parameters =  {  'rf' :
                           {'n_estimators':      [10, 50, 100, 200],
                           'min_samples_split': [2, 3],
                           'min_samples_leaf':  [1, 2, 3],
                           'n_jobs':            [MAX_JOBS],
                           'random_state':      [rnd]
                           },
                        'xgb' :
                           {
                           'min_child_weight':  [1, 5, 10],
                           'gamma':             [0.5, 1, 1.5, 2],
                           'n_jobs':            [MAX_JOBS],
                           'random_state':      [rnd],
                           'max_depth':         [3, 4, 5]
                           }
                        }
         parameters_grid = parameters[classifier]
         print('Using target: ',target_col)
         print('Using random seed: ',rnd_seed)
         df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df_X, df_y[target_col], test_size=0.3, random_state=rng)
         print('Tuning model...')
         if (np.shape(df_X_train)[0]<2000):
            class_model = ccshap.get_CV_model(df_X_train, df_y_train, model_type, parameters_grid, n_folds=5)   
         else:
            class_model = ccshap.get_CV_model_opt(df_X_train, df_y_train, model_type, parameters_grid, n_folds=5)
         print('Done.')
         output_file_name = "dataset_"+str(id_dataset)+"_"+target_col.replace(" ", "_")+"_"+str(rnd_seed)
         output_file = os.path.join(output_path, output_file_name)
         ccshap.save_model(class_model,output_file)

def main(argv):
   output_path = '.'
   id_dataset = 53
   classifier = 'xgb'
   warnings.filterwarnings('ignore') 
   opts, args = getopt.getopt(argv,"hi:o:",["help","out=","id=","params=","classifier="])
   for opt, arg in opts:
      if opt in ("-h", "--help"):
         print ('ccshap_expe.py --out <outputdir> []')
         sys.exit()
      elif opt in ("-o", "--out"):
         output_path = arg
      elif opt in ("-c", "--classifier"):
         if not(arg in ['xgb','rf']):
            sys.exit('Error: classifier model type not supported.')
         classifier = arg
      elif opt in ("-i", "--id"):
         id_dataset = int(arg)
   print("CCSHAP: grid search using "+classifier+" on: "+str(id_dataset))
   if (classifier=='xgb'):
      ccshap_full(id_dataset, output_path, classifier, model_type=XGBClassifier(verbosity=0, tree_method='hist', device='cuda'))
   elif (classifier=='rf'):
      ccshap_full(id_dataset, output_path, classifier, model_type=RandomForestClassifier(verbose=0))
if __name__ == "__main__":
   main(sys.argv[1:])