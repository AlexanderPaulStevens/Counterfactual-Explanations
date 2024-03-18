from carla import Data, DataCatalog
import pandas as pd
from typing import List
from util.DatasetManager import DatasetManager
from util.arguments import Args
from util.settings import global_setting
import numpy as np 

class OwnCatalog(DataCatalog):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset_manager = DatasetManager(dataset_name)
        self.data = self.dataset_manager.read_dataset() #read the original data (to check the max prefix length, but we might remove this)
        self.train_data, self._column_names, self.train_label = self.dataset_manager.read_preprocessed_datasets(dataset_name, 'train')
        self.test_data, _, self.test_label = self.dataset_manager.read_preprocessed_datasets(dataset_name, 'test')
        self._vocab_size = len(self._column_names)+2 # padding value and EoS token
        self.arguments = Args(dataset_name)
        self.cls_encoder_args, self.min_prefix_length, self._max_prefix_length, self.activity_col = self.arguments.extract_args(self.data, self.dataset_manager)
        print('the dataset has min prefix length of', self.min_prefix_length, 'and max prefix length of', self.max_prefix_length)
        #max_prefix_length = self.data.groupby('Case ID')['Activity'].transform(len).max()
        #self._max_prefix_length = max_prefix_length
        self.train_ratio = global_setting['train_ratio']
        self.event_names = [f"event {i}" for i in range(1, self._max_prefix_length)]
        self.catalog = {
            "categorical": self.event_names,  # List of categorical features
            "continuous": [],   # List of continuous features
            "immutables": [],   # List of immutable features
            "activity_column_names": self._column_names,
            "target_train": np.array(self.train_label),    # label of the train data
            "target_test": np.array(self.test_label),    # label of the test data
        }
        # This is calling the constructor of the parent class DataCatalog. 
        # This is to ensure that the initialization logic of the parent class is executed before adding any additional logic specific to the subclass.
        super().__init__(
            self.dataset_name,
            self.train_data,
            self.test_data,
            scaling_method = 'Identity',
            encoding_method = 'SequenceEncoding',
        )

    @property
    def categorical(self) -> List[str]:
        return self.catalog["categorical"]

    @property
    def target_train(self) -> str:
        return self.catalog["target_train"]
    
    @property
    def target_test(self) -> str:
        return self.catalog["target_test"]

    @property
    def df_train(self):
        return self.train_data

    @property
    def df_test(self):
        return self.test_data
    
    @property
    def vocab_size(self):
        return self._vocab_size
    
    @property
    def max_prefix_length(self):
        return self._max_prefix_length
    
    @property
    def name(self):
        return self.dataset_name