from util.arguments import Args
from util.DatasetManager import DatasetManager
from util.settings import global_setting
import pandas as pd
import numpy as np
import h5py
import os 

datasets = ['production','sepsis_cases_1', 'sepsis_cases_2', 'sepsis_cases_4', 
            'bpic2012_accepted', 'bpic2012_declined', 'bpic2012_cancelled', 
            'bpic2015_1_f2','bpic2015_2_f2','bpic2015_3_f2','bpic2015_4_f2','bpic2015_5_f2']
datasets = ['sepsis_cases_4']
train_ratio = global_setting['train_ratio'] #the train ratio for the train test split
generate_prefixes = False #if you want to generate prefixes or not
agg_encoding = False #if you want to use the aggregated encoding or not

for dataset_name in datasets:
    print('Dataset:', dataset_name)
    #the dataset manager to load, split and do the preprocessing for the event logs. 
    dataset_manager = DatasetManager(dataset_name)
    data = dataset_manager.read_dataset() #read the data

    #to extract the necessary arguments of the dataset. This works for all the event logs.
    arguments = Args(dataset_name)  

    #split the data, and generate cases that within a specific range
    train, test = dataset_manager.split_data_strict(data, train_ratio, split="temporal")

    # Assuming 'train' is your DataFrame
    filtered_data = train[train['label'] == 'regular']
    last_event = filtered_data.groupby('Case ID')['event_nr'].idxmax()
    result = filtered_data.loc[last_event, ['Case ID', 'Activity']]

    # 'result' now contains the last 'Activity' for each case with the label 'regular'
    print(set(result['Activity']))

    # Assuming 'train' is your DataFrame
    filtered_data = train[train['label'] == 'deviant']
    last_event = filtered_data.groupby('Case ID')['event_nr'].idxmax()
    result = filtered_data.loc[last_event, ['Case ID', 'Activity']]

    if generate_prefixes:
        # prefix generation of data
        cls_encoder_args, min_prefix_length, max_prefix_length, activity_col = arguments.extract_args(data, dataset_manager)
        print('prefix length from', min_prefix_length, 'until', max_prefix_length)
        train = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length)
        test = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length) 
    else:
        cls_encoder_args, min_prefix_length, max_prefix_length, activity_col = arguments.extract_args(data, dataset_manager)
        train, test = dataset_manager.generate_cases_of_length(train, test, cls_encoder_args, min_prefix_length, max_prefix_length)
    
    cols = [activity_col, cls_encoder_args['case_id_col'], 'label', 'event_nr', 'case_length']#what are the columns
    train = train[cols].copy()
    test = test[cols].copy()

    
    #test = dataset_manager.undersample_cases(test) #undersample the test cases
    sepsis_datasets = ['sepsis_cases_1', 'sepsis_cases_2', 'sepsis_cases_4']
    if dataset_name in sepsis_datasets:
        train = dataset_manager.preprocessing_dataset(train)
        test = dataset_manager.preprocessing_dataset(test)

    #if dataset_name in ['sepsis_cases_1']:
    #    train = dataset_manager.oversample_cases(train) # otherwise we have majority prediction
    train_iBCM, test_iBCM = train.copy(), test.copy()
    
    # Here, we use an own created class ColumnEncoder to do the encoding.   
    train_activity, test_activity, ce = dataset_manager.prepare_inputs(train_iBCM.loc[:,[activity_col]], test_iBCM.loc[:,[activity_col]]) # We can reverse (or extract the encoding mappings) with the variable 'ce'
    train_iBCM[[activity_col]], test_iBCM[[activity_col]] = train_activity, test_activity  

    dfsdf
    """
    This was to check which cases do not have the activity 'ER triage'. There are only 3, and they have the 'ER sepsis triage activity by itself'
    
    # Group the filtered DataFrame by 'Case ID' and count the number of occurrences for each activity
    case_id_counts = train_iBCM.groupby(['Case ID'])['Activity'].unique()
    cases_to_check = []
    for case_id, unique_activities in case_id_counts.items():
        if 5 not in unique_activities:
            cases_to_check.append(case_id)
            print(f"Case ID: {case_id}, Index: {case_id_counts.index.get_loc(case_id)}, Activites: {unique_activities}")

    print(cases_to_check)
    print(train[train['Case ID']=='IC'])
    print(train_iBCM[train_iBCM['Case ID']=='IC'])
    print(train[train['Case ID']=='VR'])
    print(train_iBCM[train_iBCM['Case ID']=='VR'])
    print(train[train['Case ID']=='VW'])
    print(train_iBCM[train_iBCM['Case ID']=='VW'])
    """
    ans_train_act, label_list_train_act, case_ids_train = dataset_manager.groupby_caseID(train_iBCM, cols, activity_col) # groupby case ID
    ans_test_act, label_list_test_act, case_ids_test = dataset_manager.groupby_caseID(test_iBCM, cols, activity_col)

    # Binary values for the labels
    label_lists_train = [int(1) if word == 'deviant' else int(0) for word in label_list_train_act]
    label_lists_test = [int(1) if word == 'deviant' else int(0) for word in label_list_test_act]

    if agg_encoding:
        train_final, feature_combiner, names = dataset_manager.transform_data_train(train, [activity_col], cls_encoder_args)
        test_final = dataset_manager.transform_data_test(test, feature_combiner)

        # Create a DataFrame to store the column names
        column_names_df = pd.DataFrame(data=[names])

        # Specify the CSV file paths
        file_train_csv = "./labeled_logs_csv_processed/csv_files/" + dataset_name + "/"
        file_path_test_csv = "./labeled_logs_csv_processed/csv_files/" + dataset_name + "/" 
        dataset_manager.ensure_path(file_train_csv)
        dataset_manager.ensure_path(file_path_test_csv)
        file_path_train = os.path.join(file_train_csv, "train.csv")
        file_path_test  = os.path.join(file_path_test_csv, "test.csv")
        files_dictionary_csv = {} 
        files_dictionary_csv[file_path_train] = [train_final]
        files_dictionary_csv[file_path_test] = [test_final]

    else:   
        #print(ans_train_act)
        padded_activity_train, train_final = dataset_manager.ohe_cases(ans_train_act, max_prefix_length)
        padded_activity_test, test_final = dataset_manager.ohe_cases(ans_test_act, max_prefix_length)

        # This is just to check that everything is running as it should be. 
        # I can later also use this function to go back to padded sequences
        #reversed_activity_train = dataset_manager.reverse_ohe_to_padded_activity(train_final)
        #assert padded_activity_train == reversed_activity_train
        #assert padded_activity_train == reversed_activity_train

        label_list_train_act = np.array(label_lists_train.copy())
        label_list_test_act = np.array(label_lists_test.copy())

        """
        this was to randomly shuffle the traces, otherwise your model gets batches of the same label each time
        """
        # Create a random permutation of indices
        #permutation_train = np.random.permutation(len(label_list_train_act))
        #permutation_test = np.random.permutation(len(label_list_test_act))
        # Shuffle the training data and labels using the same permutation
        #train_final = train_final[permutation_train]
        #test_final = test_final[permutation_test]
        #label_list_train_act = label_list_train_act[permutation_train]
        #label_list_test_act_shuffled = label_list_test_act[permutation_test]

        label_list_train_act = label_list_train_act.tolist()
        label_list_test_act = label_list_test_act.tolist()

        #extract the column names from the columnencoder
        column_names = list(list(ce.get_maps().values())[0].keys())

        # Create a DataFrame to store the column names
        column_names_df = pd.DataFrame(data=[column_names])

        # Create a DataFrame to store the column names
        caseids_df_train = pd.DataFrame(data=[case_ids_train])
        caseids_df_test = pd.DataFrame(data=[case_ids_test])

        # Specify the HDF5 file paths
        file_train_hdf5 = "./labeled_logs_csv_processed/hdf5_files/" + dataset_name + "/"
        file_path_test_hdf5 = "./labeled_logs_csv_processed/hdf5_files/" + dataset_name + "/" 
        dataset_manager.ensure_path(file_train_hdf5)
        dataset_manager.ensure_path(file_path_test_hdf5)

        file_path_train = os.path.join(file_train_hdf5, "train.h5")
        file_path_test  = os.path.join(file_path_test_hdf5, "test.h5")
        files_dictionary_csv = {}
        files_dictionary_csv[file_path_train] = [train_final, label_list_train_act, column_names_df, caseids_df_train]
        files_dictionary_csv[file_path_test] = [test_final, label_list_test_act, column_names_df, caseids_df_test]
        

    file_train_iBCM = "./labeled_logs_csv_processed/dat_lab_files/" + dataset_name + "/" # specify the .dat and .lab file paths
    file_test_iBCM = "./labeled_logs_csv_processed/dat_lab_files/" + dataset_name + "/" 

    dataset_manager.ensure_path(file_train_iBCM) # This is to make sure that you do not have to make folders manually (only once needed)
    dataset_manager.ensure_path(file_test_iBCM)

    # These are the actual outfiles
    file_path_train_iBCM = os.path.join(file_train_iBCM, "train.dat")
    file_path_test_iBCM = os.path.join(file_test_iBCM, "test.dat")
    labels_file_path_train_iBCM  = os.path.join(file_train_iBCM, "train.lab")
    labels_file_path_test_iBCM  = os.path.join(file_test_iBCM, "test.lab")
    
    files_dictionary_dat = {}
    files_dictionary_dat[file_path_train_iBCM] = ans_train_act
    files_dictionary_dat[file_path_test_iBCM] = ans_test_act
    files_dictionary_lab = {}
    files_dictionary_lab[labels_file_path_train_iBCM] = label_lists_train
    files_dictionary_lab[labels_file_path_test_iBCM] = label_lists_test

    #write the files to a .dat and a .lab file
    dataset_manager.write_files(files_dictionary_dat, files_dictionary_lab, files_dictionary_csv, agg_encoding)
        
# File has been written
print(f'Data and labels have been saved')

#[3, 5, 4, 7, 2, 8, 9, 6, 1, 5, 9, 2, 2, 16]