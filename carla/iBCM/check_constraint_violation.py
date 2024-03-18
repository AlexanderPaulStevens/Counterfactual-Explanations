import pandas as pd
import re 
import numpy as np
import torch
import torch.nn as nn

class ProcessConstraints:
    # Absence Constraint ('absence'): This constraint is added when an activity is absent within a trace.
    # Exactly Constraint ('exactly'): This constraint is added when there is exactly one occurrence of an activity
    # Exactly2 Constraint ('exactly2'): This constraint is added when there are exactly two occurrences of an activity
    # Existence3 Constraint ('existence3'): This constraint is added when there are three or more occurrences of an activity
    # Init Constraint ('init'): This constraint is added when the first occurrence of an activity is at the beginning of the trace.
    # Last Constraint ('last'): This constraint is added when the last occurrence of an activity is at the end of the trace.
    # Co-existence Constraint ('co_existence'): This constraint is added when activities A and B co-exist within a trace

    # precedence: When activity A precedec activity B
    # alternate precedence: when an alternating pattern of A and B occur
    # chain precedence: when there is a chain-like sequence of occurences
    # response: when A is followed by B
    # alternate response: when an alternating pattern of A and B occur
    # chain response: if there is a sequence of occurrences where A is followed by B, and this sequence forms a chain in response to each occurrence of A.
    # succession: both precedence and response conditions are met, indicating a succession relationship between A and B
    # alternate succession: when both alternate precedence and alternate response conditions are met, indicating an alternating succession relationship between A and B
    # chain succession: when both chain precedence and chain response conditions are met, indicating a chain-like succession relationship between A and B

    def __init__(self, dataset_name, vocab_size, target_class):
        self.pattern = r'\d+' # Define a regular expression pattern to match any number
        self.filename_label_specific = 'carla/iBCM/' + dataset_name + '/' + dataset_name + '_label_specific_table' + '.csv'
        self.filename_constraint = 'carla/iBCM/' + dataset_name + '/' + dataset_name + '_constraint_file' + '.txt'
        self.vocab_size = vocab_size
        self.device = "cpu"

        # read joint constraints
        with open(self.filename_constraint) as f:
            lines = f.readlines()
        # initialize the joint constraints
        self.joint_constraints = lines[0].split()

        #read label specific constraints
        fold_train_results = pd.read_csv(self.filename_label_specific)
        fold_train_results_label = fold_train_results[fold_train_results.label==target_class[0]] #take the label specific rows
        selected_columns = fold_train_results_label.columns[(fold_train_results_label == 1).all()] # Extract columns with only 1 (not zero) values. This gives you the constraints for a specific label
        self.total_mined_constraints = list(selected_columns)
        print('total constraints', self.total_mined_constraints)
        if target_class[0] == 1:
            self.total_mined_constraints.remove('label')
        self.process_loss = torch.nn.CrossEntropyLoss()
        self.relu = torch.nn.ReLU()
       
        print('the label-specific constraints are:', self.total_mined_constraints)
        print('the joint constraints are:', self.joint_constraints)

        self.total_mined_constraints.extend(self.joint_constraints)

        self.activities = list(range(vocab_size))
        self.unary_constraints = ['absence', 'exactly', 'exactly2', 'existence3', 'init', 'last', 'co_existence']
        self.binary_constraints = ['not_succession', 'precedence', 'alternate_precedence', 'chain_precedence', 'response',
                                  'alternate_response', 'chain_response', 'succession', 'alternate_succession', 'chain_succession']
        self.verbose = False

    def mine_binaries(self, act_a, act_b, a_list, b_list):
        local_constraints = []
        p=ap=r=ar=cp=cr= False
        
        if b_list[0] > a_list[0]:
            if a_list[len(a_list)-1] < b_list[0]:
                local_constraints.append(('not_succession'+"("+str(act_b)+"-"+str(act_a)+")"))
            local_constraints.append(('precedence'+"("+str(act_a)+"-"+str(act_b)+")"))
            p = True
            index = 0
            previous = next_i = b_list[0]
            go_on = True
            chain = (b_list[0] - 1) in a_list
            while go_on and (index+1) < len(b_list):
                index += 1
                next_i = b_list[index]
                if (next_i - previous) > 1:
                    for i in range(previous+1,next_i):
                        if i in a_list:
                            go_on = True
                            if (next_i-1) not in a_list:
                                chain = False
                            break
                        go_on = False
                    previous = next_i
                else:
                    go_on = False
            if next_i == b_list[len(b_list)-1] and go_on:
                if len(b_list) > 1:
                    local_constraints.append(('alternate_precedence'+"("+str(act_a)+"-"+str(act_b)+")"))
                    ap = True
                if chain:
                    local_constraints.append(('chain_precedence'+"("+str(act_a)+"-"+str(act_b)+")"))
                    cp = True
                
        if b_list[len(b_list)-1] > a_list[len(a_list)-1]:
            local_constraints.append(('response'+"("+str(act_a)+"-"+str(act_b)+")"))
    
            r = True
            
            index = 0
            go_on = True
            previous = next_i = a_list[0]
            chain = (a_list[len(a_list)-1]+1) in b_list
            while go_on and (index+1) < len(a_list):
                index += 1
                next_i = a_list[index]
                if (next_i - previous) > 1:
                    for i in range(previous+1,next_i):
                        if i in b_list:
                            go_on = True
                            if (previous+1) not in b_list:
                                chain = False
                            break
                        go_on = False
                    previous = next_i
                else:
                    go_on = False
            if next_i == a_list[len(a_list)-1] and go_on:
                if len(a_list) > 1:
                    local_constraints.append(('alternate_response'+"("+str(act_a)+"-"+str(act_b)+")"))
                    ar = True
                if chain:
                    local_constraints.append(('chain_response'+"("+str(act_a)+"-"+str(act_b)+")"))
                    cr = True
                        
        if p and r:
                local_constraints.append(('succession'+"("+str(act_a)+"-"+str(act_b)+")"))            
        if ap and ar:
                local_constraints.append(('alternate_succession'+"("+str(act_a)+"-"+str(act_b)+")"))
        if cp and cr:
                local_constraints.append(('chain_succession'+"("+str(act_a)+"-"+str(act_b)+")"))
        return local_constraints
        
    def check_if_constraints_are_met(self, tensor, constraint, act_positions, activity1, activity2):
        if 'init' in constraint:
            condition = act_positions[activity1]
        
            if not int(0) in condition:
                self.violation_counter +=1  
        
        elif 'absence' in constraint:
            condition = act_positions[activity1]
          
            if len(condition) >0:
                self.violation_counter +=1
            
        elif 'exactly' in constraint:  
            condition = act_positions[activity1]
        
            if len(condition) != 1:
                self.violation_counter +=1

        elif 'exactly2' in constraint:
            condition = act_positions[activity1]
     
            if len(condition) != 2:
                self.violation_counter +=1

        elif 'existence3' in constraint:
            condition = act_positions[activity1]
    
            if len(condition) != 3:
                self.violation_counter +=1

        # voeg hier exclusive choice toe
        elif 'co_existence' in constraint:
            condition1 = len(act_positions[activity1])>0
            condition2 = len(act_positions[activity2])>0
            if not(activity1 != activity2 and condition1 and condition2):
                self.violation_counter +=1
        
        elif 'last' in constraint:
            # Find the last row where the value 1 is not in the first row
            for row in range(len(tensor) - 1, -1, -1): # Iterate through rows in reverse order
                if tensor[row, 0] != 1: # The first column value is not 1
                    end_activity = torch.nonzero((tensor[row] == 1))[0].item()
                    #.squeeze(dim=1).item() # Find the column number where the value is 1
                    break  # Exit the loop after finding the first matching row
            if activity2 != end_activity:
                self.violation_counter +=1

    def find_violated_traces(self, tensor, constraint_list = None, verbose=False):
        indices_to_keep = []
        index = 0
        for i in tensor:
            violation_counts, constraints = self.count_violations(i, constraint_list, verbose)
            if verbose:
                print('the violations for trace:')
                print(torch.argmax(i,dim=1))
                print(violation_counts, len(constraints))
                print(violation_counts/len(constraints)*100, '%')
            if violation_counts>0:
                indices_to_keep.append(index)
            index +=1
        if len(indices_to_keep)>0:
            selected_tensors = torch.stack([tensor[i] for i in indices_to_keep])
            return selected_tensors
        else:
            return
    
    def init_constraint(self, activity, reconstruction):
        init_act = torch.tensor([activity]*reconstruction.shape[0]).to(self.device)
        init_loss = self.process_loss(reconstruction[:,0,:], init_act).to(self.device)
        return init_loss

    def calculate_joint_constraint_loss(self, tensor_batch):
        loss = 0
        tensor_batch = tensor_batch.to(self.device)
        for constraint in self.joint_constraints:
            numbers = re.findall(self.pattern, constraint) # Use re.findall to find all the activity numbers in the constraint
            activity1 = int(numbers[0])
            if 'init' in constraint:
                loss += self.init_constraint(activity1, tensor_batch).to(self.device)
        # for each tensor in the batch, check the violation of the joint constraints
        for i in tensor_batch:
            i = i.to(self.device)
            violations, _ = self.count_violations(i, 'joint')
            loss += violations
        return loss
 
    def compute_label_violation_loss(self, tensor):
        total_weight = 0
        """
        for constraint in self.mined_constraints: #for each constraint, you check whether your tensor is complaint with it
            #print('constraint', constraint)
            iter+=1
            if 'label' in constraint:
                break
            numbers = re.findall(self.pattern, constraint) # Use re.findall to find all the activity numbers in the constraint
            activityA = int(numbers[0])
            activityB = int(numbers[1])
            if self.verbose:
                print('activityA', activityA, 'activityB', activityB)
            
            print(abs(torch.max(tensor[:,activityA])))
            #if activityA == activityB:
            #    total_weight -= abs(torch.max(tensor[:,activityA]))
            #else:
            #    total_weight -= abs(torch.max(tensor[:,activityA]))
            #    total_weight -= abs(torch.max(tensor[:,activityB]))
            
            if 'succession' in constraint:
                #print(constraint)
                #total_weight += abs(tensor[0,activityB]) #you add this weight to the loss. The second activity can never be in the first place of the sequence

                _, indexes  = torch.max(tensor, dim=1)
                indices_activity_A = (indexes == activityA).nonzero()
                indices_activity_B = (indexes == activityB).nonzero()
                if indices_activity_A.shape==(0, 1):
                    total_weight += 10
                    total_weight += 10
                elif indices_activity_B.shape==(0,1):
                    total_weight += 10

                else:
                    # Calculate the violations based on element-wise comparison
                    violations = (indices_activity_B > indices_activity_A).float()
                    print('violations', violations)

                    # Count the number of violations
                    total_weight += torch.sum(violations)
                if self.verbose:
                    print('total weight after total sum', total_weight)
            """
        return total_weight

    def count_violations(self,tensor, constraint_list = None, verbose = False): 
        self.violation_counter = 0
        act_positions = {} # {activity: [positions]}
        tensor_check = tensor.clone().detach().to(self.device)
        # Find the column index with the highest probability for each row
        argmax_indices = torch.argmax(tensor_check, dim=1).to(self.device)
        # Assuming tensor_check and argmax_indices are already defined tensors
        one_hot_encoded = torch.zeros_like(tensor_check).to(self.device)
        one_hot_encoded[torch.arange(len(argmax_indices)), argmax_indices] = 1
        tensor_check = one_hot_encoded.to(self.device)
        if constraint_list == 'total':
            given_constraints = self.total_mined_constraints
        elif constraint_list =='joint':
            given_constraints = self.joint_constraints
        for a in self.activities:
            condition = tensor_check[:,a] == 1 # Check in which row the activity is being used
            true_indices = [index for index, value in enumerate(condition) if value]
            act_positions[a] = true_indices

        for constraint in given_constraints: #for each constraint, you check whether your tensor is complaint with it
            numbers = re.findall(self.pattern, constraint) # Use re.findall to find all the activity numbers in the constraint
            activity1 = int(numbers[0])
            activity2 = int(numbers[1])
            local_constraints = []
            if verbose:
                print('activities', activity1, activity2)
                print('constraint', constraint)

            if any(total in constraint for total in self.unary_constraints):
                self.check_if_constraints_are_met(tensor_check, constraint, act_positions, activity1, activity2)
                if verbose:
                    print(self.violation_counter)

            elif any(total in constraint for total in self.binary_constraints):
                #print('the violation counter', self.violation_counter)
                a_list = act_positions[activity1]
                b_list = act_positions[activity2]

                if len(a_list)==0 or len(b_list)==0:
                    self.violation_counter +=1
                    if verbose:
                        print(self.violation_counter)
                else:
                    local_constraints.extend(self.mine_binaries(activity1, activity2, a_list, b_list))
                    local_constraints.extend(self.mine_binaries(activity2, activity1, b_list, a_list))

                    
                    if constraint not in local_constraints:
                       
                       self.violation_counter +=1
                    if verbose:
                        print('activity 1', 'activity2', a_list, b_list)
                        print('local constraints', local_constraints)
                        print(self.violation_counter)
                        
           
        return self.violation_counter, given_constraints