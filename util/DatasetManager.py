import os
import util.dataset_confs as dataset_confs
import util.EncoderFactory as EncoderFactory
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from collections import OrderedDict
from pandas.api.types import is_string_dtype
import h5py
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MinMaxScaler
import torch

class DatasetManager:

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

        self.case_id_col = dataset_confs.case_id_col[self.dataset_name]
        self.activity_col = dataset_confs.activity_col[self.dataset_name]
        self.timestamp_col = dataset_confs.timestamp_col[self.dataset_name]
        self.label_col = dataset_confs.label_col[self.dataset_name]
        self.pos_label = dataset_confs.pos_label[self.dataset_name]
        self.encoding_dict = {"agg": ["agg"]}
        self.cls_encoding ='agg'
        self.dynamic_cat_cols = dataset_confs.dynamic_cat_cols[self.dataset_name]
        self.static_cat_cols = dataset_confs.static_cat_cols[self.dataset_name]
        self.dynamic_num_cols = dataset_confs.dynamic_num_cols[self.dataset_name]
        self.static_num_cols = dataset_confs.static_num_cols[self.dataset_name]
        self.sorting_cols = [self.timestamp_col, self.activity_col]

    def read_dataset(self):
        # read dataset
        dtypes = {col: "object" for col in self.dynamic_cat_cols+self.static_cat_cols+[self.case_id_col, self.label_col, self.timestamp_col]}
        for col in self.dynamic_num_cols + self.static_num_cols:
            dtypes[col] = "float"

        data = pd.read_csv(dataset_confs.filename[self.dataset_name], sep=";", dtype=dtypes)
        data[self.timestamp_col] = pd.to_datetime(data[self.timestamp_col])

        return data

    def read_preprocessed_datasets(self, dataset_name, mode='None'):
        # read preprocessed dataset
        if mode=='train':
            # These are the actual outfiles
            outfile = "./labeled_logs_csv_processed/hdf5_files/" + dataset_name + "/train.h5"
        elif mode== 'test':
            outfile = "./labeled_logs_csv_processed/hdf5_files/" + dataset_name + "/test.h5" 
   
         #Load training data again, to see if it works.
        with h5py.File(outfile, 'r') as hdf5_file:
            # Load the column names DataFrame from the attribute
            loaded_column_names_df = pd.read_json(hdf5_file.attrs['column_names'])
            loaded_column_names = loaded_column_names_df.values[0].tolist()
            # Load the label array
            loaded_label = np.array(hdf5_file['labels'])
            # Load the dataframe
            loaded_data = torch.LongTensor(hdf5_file['data'])

        return loaded_data, loaded_column_names, loaded_label

    def split_data(self, data, train_ratio, split="temporal", seed=22):
        # split into train and test using temporal split

        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        if split == "temporal":
            start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind="mergesort")
        elif split == "random":
            np.random.seed(seed)
            start_timestamps = start_timestamps.reindex(np.random.permutation(start_timestamps.index))
        train_ids = list(start_timestamps[self.case_id_col])[:int(train_ratio*len(start_timestamps))]
        train = data[data[self.case_id_col].isin(train_ids)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
        test = data[~data[self.case_id_col].isin(train_ids)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')

        return (train, test)

    def split_data_strict(self, data, train_ratio, split="temporal"):
        # split into train and test using temporal split and discard events that overlap the periods
        data = data.sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind='mergesort')
        train_ids = list(start_timestamps[self.case_id_col])[:int(train_ratio*len(start_timestamps))]
        train = data[data[self.case_id_col].isin(train_ids)].sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        test = data[~data[self.case_id_col].isin(train_ids)].sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        split_ts = test[self.timestamp_col].min()
        train = train[train[self.timestamp_col] < split_ts]
        return (train, test)

    def split_data_discard(self, data, train_ratio, split="temporal"):
        # split into train and test using temporal split and discard events that overlap the periods
        data = data.sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind='mergesort')
        train_ids = list(start_timestamps[self.case_id_col])[:int(train_ratio*len(start_timestamps))]
        train = data[data[self.case_id_col].isin(train_ids)].sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        test = data[~data[self.case_id_col].isin(train_ids)].sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        split_ts = test[self.timestamp_col].min()
        overlapping_cases = train[train[self.timestamp_col] >= split_ts][self.case_id_col].unique()
        train = train[~train[self.case_id_col].isin(overlapping_cases)]
        return (train, test)

    def split_val(self, data, val_ratio, split="random", seed=22):
        # split into train and test using temporal split
        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        if split == "temporal":
            start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind="mergesort")
        elif split == "random":
            np.random.seed(seed)
            start_timestamps = start_timestamps.reindex(np.random.permutation(start_timestamps.index))
        val_ids = list(start_timestamps[self.case_id_col])[-int(val_ratio*len(start_timestamps)):]
        val = data[data[self.case_id_col].isin(val_ids)].sort_values(self.sorting_cols, ascending=True, kind="mergesort")
        train = data[~data[self.case_id_col].isin(val_ids)].sort_values(self.sorting_cols, ascending=True, kind="mergesort")
        return (train, val)

    def generate_prefix_data(self, data, min_length, max_length, gap=1):
        # generate prefix data (each possible prefix becomes a trace)
        case_length = data.groupby(self.case_id_col)[self.activity_col].transform(len)

        data.loc[:, 'case_length'] = case_length.copy()
        dt_prefixes = data[data['case_length'] >= min_length].groupby(self.case_id_col).head(min_length)
        dt_prefixes["prefix_nr"] = 1
        dt_prefixes["orig_case_id"] = dt_prefixes[self.case_id_col]
        for nr_events in range(min_length+gap, max_length+1, gap):
            tmp = data[data['case_length'] >= nr_events].groupby(self.case_id_col).head(nr_events)
            tmp["orig_case_id"] = tmp[self.case_id_col]
            tmp[self.case_id_col] = tmp[self.case_id_col].apply(lambda x: "%s_%s" % (x, nr_events))
            tmp["prefix_nr"] = nr_events
            dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)

        dt_prefixes['case_length'] = dt_prefixes['case_length'].apply(lambda x: min(max_length, x))

        return dt_prefixes
    
    def generate_cases_of_length(self, train, test, cls_encoder_args, min_prefix_length, max_prefix_length):
        # create the case lengths
        train['case_length'] = train.groupby(cls_encoder_args['case_id_col'])[self.activity_col].transform(len)
        test['case_length'] = test.groupby(cls_encoder_args['case_id_col'])[self.activity_col].transform(len)
        
        # filter out the short cases
        train = train[train['case_length'] >= min_prefix_length].groupby(cls_encoder_args['case_id_col']).head(max_prefix_length) #because you need to add a EoS token
        test = test[test['case_length'] >= min_prefix_length].groupby(cls_encoder_args['case_id_col']).head(max_prefix_length)

        print('prefix lengths of', min_prefix_length, 'until', max_prefix_length)
        print('there are', len(set(train[train['label']=='deviant']['Case ID'])), 'deviant cases and', len(set(train[train['label']=='regular']['Case ID'])), 'regular cases')
        return train, test
    
    def is_ordered(self, lst):
        return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))

    def groupby_caseID(self, data, cols, col):
        # Initialize lists to store sequences and labels for 'deviant' and 'regular' cases
        deviant_sequences, regular_sequences = [], [] 
        deviant_labels, regular_labels = [], []  
        groups = data[cols].groupby('Case ID', as_index=False)
        groups = groups.apply(lambda group: group.sort_values('event_nr'))
        grouped = groups.groupby('Case ID')

        case_id_deviant, case_id_regular = [], []
        for case_id, group in grouped:
            #print(f"Group for Case ID {case_id}:")
            #print('event number', list(group['event_nr']))
            if self.is_ordered(list(group['event_nr'])): 
                label = group['label'].iloc[0]
                sequence = list(group[col])
                sequence.extend([self.vocab_size-1]) # add EoS token
                if label == 'deviant':
                    deviant_sequences.append(sequence)
                    deviant_labels.append(label)
                    case_id_deviant.append(case_id)
                elif label == 'regular':
                    regular_sequences.append(sequence)
                    regular_labels.append(label)
                    case_id_regular.append(case_id)

            else:
                print('problem', group)
                breakpoint()
                AssertionError
        # Concatenate sequences and labels for 'deviant' and 'regular' cases
        sequences = regular_sequences + deviant_sequences
        label_lists = regular_labels + deviant_labels
        label_lists = regular_labels + deviant_labels
        case_ids = case_id_regular + case_id_deviant
        return sequences, label_lists, case_ids
    
    def ensure_path(self, file):
        if not os.path.exists(os.path.join(file)):
            os.makedirs(os.path.join(file))

    def undersample_cases(self, data, undersample_level=None):
        unique_cases = data.copy()
        unique_cases = unique_cases.drop_duplicates(subset=['Case ID', 'label'])

        label_counts = unique_cases['label'].value_counts() # Calculate class distribution
        print('label counts before', label_counts)

        overrepresented_label = label_counts.idxmax() # Find the labels with their counts
        underrepresented_label = label_counts.idxmin()

        num_cases_to_keep = round(label_counts.min()/undersample_level) # Count of cases to keep for each class (matching the underrepresented class)
        
        overrepresented_case_ids = data[data['label'] == overrepresented_label]['Case ID'].unique() # Randomly sample cases from the overrepresented label to match the underrepresented label
        selected_overrepresented_case_ids = np.random.choice(overrepresented_case_ids, size=num_cases_to_keep, replace=False)

        data = pd.concat([ 
            data[(data['label'] == overrepresented_label) & (data['Case ID'] == case_id)]
            for case_id in selected_overrepresented_case_ids
        ] + [
            data[data['label'] == underrepresented_label]
        ]) # Filter the DataFrame to include the sampled cases and all cases from the underrepresented label

        unique_cases = data.copy()
        unique_cases = unique_cases.drop_duplicates(subset=['Case ID', 'label'])
        label_counts = unique_cases['label'].value_counts() # Count the labels again
        print('label counts train after', label_counts)
        return data
    
    def oversample_cases(self, data):
        unique_cases = data.copy()
        unique_cases = unique_cases.drop_duplicates(subset=['Case ID', 'label'])

        label_counts = unique_cases['label'].value_counts() # Calculate class distribution
        print('label counts before', label_counts)

        overrepresented_label = label_counts.idxmax() # Find the labels with their counts
        underrepresented_label = label_counts.idxmin()

        data2 = data[data['label'] == underrepresented_label]
        data2['Case ID'] = data2['Case ID'] + 'oversampled'

        data3 = data[data['label'] == underrepresented_label]
        data3['Case ID'] = data3['Case ID'] + 'oversampled2'

        data4 = data[data['label'] == underrepresented_label]
        data4['Case ID'] = data4['Case ID'] + 'oversampled3'

        data5 = data[data['label'] == underrepresented_label]
        data5['Case ID'] = data5['Case ID'] + 'oversampled4'

        data = pd.concat([data[data['label'] == underrepresented_label]] + [data2] + [data3] + [data4] + [data5]
        + [data[data['label'] == overrepresented_label]]) # Filter the DataFrame to include the sampled cases and all cases from the underrepresented label

        unique_cases = data.copy()
        unique_cases = unique_cases.drop_duplicates(subset=['Case ID', 'label'])
        label_counts = unique_cases['label'].value_counts() # Count the labels again
        print('label counts train after', label_counts)
        return data
    

    def write_files(self, files_dictionary_dat, files_dictionary_lab, files_dictionary_hdf5, agg_encoding):
        
    
        for key in files_dictionary_dat.keys():
            # Open the file in write mode
            with open(key, 'w') as file: # Iterate through each inner list
                for inner_list in files_dictionary_dat[key]: # Convert the inner list to a string with space-separated numbers
                    row = ' '.join(map(str, inner_list))
                    file.write(row + '\n') # Write the row to the file
            
        for key in files_dictionary_lab.keys():
            # Save the labels to the .lab file
            with open(key, 'w') as labels_file:
                for label in files_dictionary_lab[key]:
                    labels_file.write(str(label) + '\n')

        if agg_encoding:
            for key in files_dictionary_hdf5.keys():
                # Create an HDF5 train file
                with open(key, 'w'):
                    # Store the dataframe to a csv file
                    files_dictionary_hdf5[key][0].to_csv(key)
                   
        else:
            for key in files_dictionary_hdf5.keys():
                # Create an HDF5 train file
                with h5py.File(key, 'w') as hdf5_file:
                        # Store the 3D array in the HDF5 file
                        hdf5_file.create_dataset('data', data=files_dictionary_hdf5[key][0])
                        # Store the labels as a dataset
                        hdf5_file.create_dataset('labels', data=files_dictionary_hdf5[key][1])
                        # Store the column names DataFrame as an attribute
                        hdf5_file.attrs['column_names'] = files_dictionary_hdf5[key][2].to_json()
                        # Store the case ids DataFrame as an attribute
                        hdf5_file.attrs['case_ids'] = files_dictionary_hdf5[key][3].to_json()

    def ohe_cases(self, activity_lists, max_prefix_length):
        # Pad activity lists with zeros to ensure uniform length
        padded_activity = [
            seq + [0] * (max_prefix_length +1 - len(seq)) if len(seq) < max_prefix_length else seq
            for seq in activity_lists
        ]

        print('padded activity', padded_activity)
        
        # Initialize an empty numpy array
        num_instances = len(padded_activity)
        one_hot_matrix = np.zeros((num_instances, max_prefix_length+1, self.vocab_size), dtype=int)
        
        # Iterate over sequences and populate the matrix
        for i, seq in enumerate(padded_activity):
            for j, activity in enumerate(seq):
                one_hot_matrix[i, j, activity] = 1
        #one_hot_matrix = one_hot_matrix[:,:,1:]
        return padded_activity, one_hot_matrix

    def reverse_ohe_to_padded_activity(self, one_hot_matrix):
        # Get the number of instances, sequence length, and vocabulary size
        num_instances, max_sequence_length, vocab_size = one_hot_matrix.shape

        # Initialize an empty list to store the reversed padded activity sequences
        reversed_activity = []
        print('here')
        # Iterate over instances and sequences
        for i in range(num_instances):
            sequence = []
            for j in range(max_sequence_length):
                # Find the index where one-hot encoding is 1
                activity_index = np.argmax(one_hot_matrix[i, j, :])
                # Append the activity index to the sequence
                sequence.append(activity_index.item())
            reversed_activity.append(sequence)

        return reversed_activity
    
    def edit_distance(self, factual, counterfactual, verbose=False) -> int:
        """
        Calculate the word level edit (Levenshtein) distance between two sequences.

        .. devices:: CPU

        The function computes an edit distance allowing deletion, insertion and
        substitution. The result is an integer.

        For most applications, the two input sequences should be the same type. If
        two strings are given, the output is the edit distance between the two
        strings (character edit distance). If two lists of strings are given, the
        output is the edit distance between sentences (word edit distance). Users
        may want to normalize the output by the length of the reference sequence.

        Args:
            seq1 (Sequence): the first sequence to compare.
            seq2 (Sequence): the second sequence to compare.
        Returns:
            int: The distance between the first and second sequences.
        """
        factual_max = np.argmax(factual, axis=1)
        counterfactual_max = np.argmax(counterfactual, axis=1)
        if verbose:
            print('edit distance', factual_max.shape, counterfactual_max.shape)
            print(factual_max, counterfactual_max)
            print('factual', factual_max, 'counterfactual', counterfactual_max)
        len_sent2 = len(counterfactual_max)
        dold = list(range(len_sent2 + 1))
        dnew = [0 for _ in range(len_sent2 + 1)]

        for i in range(1, len(factual_max) + 1):
            dnew[0] = i
            for j in range(1, len_sent2 + 1):
                if factual_max[i - 1] == counterfactual_max[j - 1]:
                    dnew[j] = dold[j - 1]
                else:
                    substitution = dold[j - 1] + 1
                    insertion = dnew[j - 1] + 1
                    deletion = dold[j] + 1
                    dnew[j] = min(substitution, insertion, deletion)

            dnew, dold = dold, dnew

        return int(dold[-1])
    
    def transform_data_train(self, dt_train, cat_cols, cls_encoder_args):
        # feature combiner and columns
        y_train = self.get_label_numeric(dt_train)
        methods = self.encoding_dict[self.cls_encoding]
        feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(
            method, case_id_col=cls_encoder_args['case_id_col'], static_cat_cols=list(), static_num_cols=list(), dynamic_cat_cols=cat_cols,
            dynamic_num_cols=list(), fillna=False, max_events=None, activity_col=cat_cols[0], timestamp_col=None,
            scale_model=None)) for method in methods])
        feature_combiner.fit(dt_train, y_train)
   
        # transform train dataset and add the column names back to the dataframe
        train_named, names = feature_combiner.transform(dt_train)
        train_named = pd.DataFrame(train_named)
        train_named.columns = names
        
        #scale dataset
        scaler = MinMaxScaler()
        #train_named_scaled = scaler.fit_transform(train_named)
        #train_named = pd.DataFrame(train_named_scaled, columns=train_named.columns)
        #self.scaler = scaler

        # add label
        train_named['label'] = y_train
   
        return train_named, feature_combiner, names

    def transform_data_test(self, dt_test, feature_combiner):
        y_test = self.get_label_numeric(dt_test)
        # transform test dataset
        test_named, names = feature_combiner.transform(dt_test)
        test_named = pd.DataFrame(test_named)
        test_named.columns = names
        #test_named_scaled = self.scaler.transform(test_named)
        #test_named = pd.DataFrame(test_named_scaled, columns=test_named.columns)

        # add label
        test_named['label'] = y_test

        return test_named
    
    def preprocessing_dataset(self, data):
        # Create a mask to identify rows to be deleted where ER registration is not the first event
        mask = (data['event_nr'] == 1) & (data['Activity'] != 'ER Registration')    
        data = data[~mask] # Invert the mask to keep rows with event_nr != 1 or 'activity' == 'ER Registration'

        #now delete the rows for the cases where there are events before the event of activity ER Registration took place
        cases_remaining_2 = data[(data.Activity =='ER Registration')&(data.event_nr.isin([3]))]['Case ID'].unique().tolist()
        mask = (data['Case ID'].isin(cases_remaining_2) & (data.event_nr==2)&(data.Activity != 'ER Registration'))
        data = data[~mask] # Invert the mask to keep rows with event_nr != 1 or 'activity' == 'ER Registration'
        
        cases_remaining_2 = data[(data.Activity =='ER Registration')&(data.event_nr.isin([4]))]['Case ID'].unique().tolist()
        mask = (data['Case ID'].isin(cases_remaining_2) & (data.event_nr==3)&(data.Activity != 'ER Registration'))
        data = data[~mask] # Invert the mask to keep rows with event_nr != 1 or 'activity' == 'ER Registration'

        mask = (data['Case ID'].isin(cases_remaining_2) & (data.event_nr==2)&(data.Activity != 'ER Registration'))
        data = data[~mask] # Invert the mask to keep rows with event_nr != 1 or 'activity' == 'ER Registration'

        excluded_events = [1, 2, 3, 4]
        filtered_cases = data[(data.Activity == 'ER Registration') & ~((data.event_nr.isin(excluded_events)))]['Case ID'].unique().tolist()
        data = data[~data['Case ID'].isin(filtered_cases)]
        return data

    def get_case_length_quantile(self, data, label, quantile=0.95):
        return int(np.ceil(data[data['label']== label].groupby(self.case_id_col).size().quantile(quantile)))

    def get_indexes(self, data):
        return data.groupby(self.case_id_col).first().index

    def get_relevant_data_by_indexes(self, data, indexes):
        return data[data[self.case_id_col].isin(indexes)]

    def get_label(self, data):
        return data.groupby(self.case_id_col).first()[self.label_col]

    def get_prefix_lengths(self, data):
        return data.groupby(self.case_id_col).last()["prefix_nr"]

    def get_case_ids(self, data, nr_events=1):
        case_ids = pd.Series(data.groupby(self.case_id_col).first().index)
        if nr_events > 1:
            case_ids = case_ids.apply(lambda x: "_".join(x.split("_")[:-1]))
        return case_ids

    def get_label_numeric(self, data):
        y = self.get_label(data)  # one row per case
        return [1 if label == self.pos_label else 0 for label in y]

    def get_class_ratio(self, data):
        class_freqs = data[self.label_col].value_counts()
        return class_freqs[self.pos_label] / class_freqs.sum()

    def get_stratified_split_generator(self, data, n_splits=5, shuffle=True, random_state=22):
        grouped_firsts = data.groupby(self.case_id_col, as_index=False).first()
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        for train_index, test_index in skf.split(grouped_firsts, grouped_firsts[self.label_col]):
            current_train_names = grouped_firsts[self.case_id_col][train_index]
            train_chunk = data[data[self.case_id_col].isin(current_train_names)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
            test_chunk = data[~data[self.case_id_col].isin(current_train_names)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
            yield (train_chunk, test_chunk)

    def get_idx_split_generator(self, dt_for_splitting, n_splits=5, shuffle=True, random_state=22):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        for train_index, test_index in skf.split(dt_for_splitting, dt_for_splitting[self.label_col]):
            current_train_names = dt_for_splitting[self.case_id_col][train_index]
            current_test_names = dt_for_splitting[self.case_id_col][test_index]
            yield (current_train_names, current_test_names)


    def prepare_inputs(self, X_train, X_test):
        global ce
        ce = ColumnEncoder()
        # converts the columns in the X_train and X_test to string types. 
        # This is typically done to ensure that all values in the DataFrame are treated as strings when encoding.
        X_train, X_test = X_train.astype(str), X_test.astype(str)
        X_train_enc = ce.fit_transform(X_train)
        X_test_enc = ce.transform(X_test)
        #extract the vocabulary size
        self.vocab_size = len(list(list(ce.get_maps().values())[0].keys()))+2 #padding and EoS token
        print('vocab size with padding value and EoS token:', self.vocab_size)
        print('dictionary of activity values', list(list(ce.get_maps().values())))
        return X_train_enc, X_test_enc, ce

# https://towardsdatascience.com/using-neural-networks-with-embedding-layers-to-encode-high-cardinality-categorical-variables-c1b872033ba2
class ColumnEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns = None
        self.maps = dict()

    def transform(self, X):
        # encodes its categorical columns using the encoding scheme stored in self.maps
        X_copy = X.copy()
        for col in self.columns:
            # encode value x of col via dict entry self.maps[col][x]+1 if present, otherwise 0
            X_copy.loc[:, col] = X_copy.loc[:, col].apply(lambda x: self.maps[col].get(x, -1))
        # It returns a copy of X with the categorical columns replaced by their corresponding integer encodings.
        return X_copy
    
    def get_maps(self):
        # This method allows you to retrieve the encoding mappings stored in 
        return self.maps

    def inverse_transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            values = list(self.maps[col].keys())
            # find value in ordered list and map out of range values to None
            X_copy.loc[:, col] = [values[i-1] if 0 < i <= len(values) else None for i in X_copy[col]]
        return X_copy

    def fit(self, X, y=None):
        # only apply to string type columns
        # This method is called during the fitting process. 
        # It identifies the categorical columns in the input DataFrame X
        # Stores them in self.maps
        # These mappings are dictionaries that map each unique categorical value to an integer
        self.columns = [col for col in X.columns if is_string_dtype(X[col])]
        for col in self.columns:
            self.maps[col] = OrderedDict({value: num+1 for num, value in enumerate(sorted(set(X[col])))})
        return self

