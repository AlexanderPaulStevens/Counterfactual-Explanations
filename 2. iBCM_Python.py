import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.naive_bayes import MultinomialNB as NB
from sklearn.tree import DecisionTreeClassifier as DT
from util.DatasetManager import DatasetManager
from carla.iBCM.run_iBCM import iBCM, iBCM_verify

def run_iBCM(dataset_name, support): 
    ## Read files
     # Specify the .dat and .lab file paths
    file_train = "./labeled_logs_csv_processed/dat_lab_files/" + dataset_name + "/"

    file_path_train = os.path.join(file_train, "train.dat")
    labels_file_path_train  = os.path.join(file_train, "train.lab")
    trace_file_train = open(file_path_train, 'r')
    label_file_train = open(labels_file_path_train, 'r')

    #####################################
    traces = []
    label_list = []
    for trace, label in zip(trace_file_train, label_file_train):
        traces.append(trace)
        label_list.append(label.replace('\n',''))
   
    label_set = set(label_list)
    no_labels = len(label_set)
    print('#labels:',no_labels)
    print("#traces", len(traces))

    dataset_manager = DatasetManager(dataset_name)
    folder_name_dataset =  'carla/iBCM/' + dataset + '/'
    dataset_manager.ensure_path(folder_name_dataset)
    # Specify the file name
    filename_joint_constraints = folder_name_dataset + dataset  +'_constraint_file' + '.txt'
    final_constraints = iBCM(filename_joint_constraints, traces, label_list, reduce_feature_space, support)   
    for constraint in final_constraints:
        print('const', constraint)

    ########################## 
    filename_train = folder_name_dataset + dataset +'_label_specific_table' + '.csv'
    
    # Label training data
    iBCM_verify(filename_train, traces, label_list, final_constraints)
        
    # Label test data
    #iBCM_verify(filename_test, traces_test, label_list_test, final_constraints)
    #fold_test_results = pd.read_csv(filename_test)
                
    #os.remove(filename_train)
    #os.remove(filename_test)
         
    #######################
    
### Start program and enter parameters
reduce_feature_space = True
datasets = ['sepsis_cases_1', 'sepsis_cases_2', 'sepsis_cases_4', 
    'production', 'bpic2012_declined', 'bpic2012_cancelled', 'bpic2012_accepted', 
    'bpic2015_1_f2','bpic2015_2_f2','bpic2015_3_f2','bpic2015_4_f2','bpic2015_5_f2']

for dataset in ['bpic2015_5_f2']:
    print('\nDataset:', dataset)
    for support in [1]:#0.2,0.4,0.6,0.8]:
        print('\nSupport level:', support)
        run_iBCM(dataset, support)