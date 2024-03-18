from util.settings import global_setting, model_setting, training_setting

class Args:
    """preprocessing for the machine learning models."""

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def extract_args(self, data, dataset_manager):
        cls_encoder_args = {'case_id_col': dataset_manager.case_id_col, 
                        'static_cat_cols': [],
                        'static_num_cols': [],
                        'dynamic_cat_cols': ["Activity"],
                        'dynamic_num_cols': [],
                        'fillna': True}
        
        # determine min and max (truncated) prefix lengths
        min_prefix_length = 1
        #if "traffic_fines" in self.dataset_name:
        #    min_prefix_length = 2

        #elif "bpic2017" in self.dataset_name:
        #    min_prefix_length = 10

        #elif "bpic2012" in self.dataset_name:
        #    min_prefix_length = 15

        #elif "production" in self.dataset_name:
        #    min_prefix_length = 1

        #elif "bpic2015" in self.dataset_name:
        #    max_prefix_length = 40
        #elif "sepsis_cases_1" in self.dataset_name:
        #    min_prefix_length = 4

        #elif "sepsis_cases_2" in self.dataset_name:
        #    min_prefix_length = 4
        case_length = data.groupby(dataset_manager.case_id_col)[dataset_manager.activity_col].transform(len)
        print('the minimum prefix length in the dataset is', case_length.min())
        min_prefix_length = max(min_prefix_length, case_length.min())
        max_prefix_length = dataset_manager.get_case_length_quantile(data, 'regular', 0.90)
        
        #max_prefix_length = data.groupby('Case ID')['Activity'].transform(len).max()
        activity_col = [x for x in cls_encoder_args['dynamic_cat_cols'] if 'Activity' in x][0]
        return cls_encoder_args, min_prefix_length, max_prefix_length, activity_col