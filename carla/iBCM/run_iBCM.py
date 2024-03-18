from carla.iBCM.ConstraintMiner import ConstraintMining, Annotated_trace, reduce_feature_space
import pandas as pd

VERBOSE = True

def iBCM(filename, traces, labels, rfs, min_sup=0.01):   
    """This function is used for constraint mining. It loads the datasets and mines the constraints based.
    """
    #filename: A filename for saving constraints
    #traces: A list of traces or sequences from the dataset.
    #labels: A list of labels associated with each trace.
    #rfs: A boolean flag indicating whether feature reduction should be performed.
    #min_sup: A minimum support threshold for constraint mining (default is 0.01).
    print('\nRunning iBCM training')
    
    #this set is to store the discovered constraints
    constraints = set()
      
    constraints_per_label, annotated_traces = load_dataset_mine_constraints(traces, labels, min_sup)
    #for i in annotated_traces:
    #        for j in annotated_traces[i]:
    #            print(j.print_constraints())

    trace = traces[0]
    trace = trace.replace('\n', '')

    print('\n**********************\nFinal stats: ')
    
    all_constraints = set()
    for label, constraints in constraints_per_label.items():
        print('Label ', label, ' has ', len(constraints), ' constraints')
        #print('the constraints are:')
        #for i in constraints:
        #    print('constraint', i)
        all_constraints = all_constraints.union(set(constraints))
        # Convert the dictionary keys to a constraints_list = list(constraints)

    print('\nTotal #constraints: ', len(all_constraints))
    joint_features = set()
    for constraint in all_constraints:
        contained = True
        for label, constraints in constraints_per_label.items():
            if constraint not in constraints:
                contained = False
                break
        if contained:
            joint_features.add(constraint)
    
    all_constraints = all_constraints.difference(joint_features)
                
    if rfs:
        final_constraints = reduce_feature_space(all_constraints)
    else:
        final_constraints = all_constraints
    print('Final total #constraints: ', len(final_constraints))

    feature_list = []
    for i in final_constraints:
        feature_list.append(str(i))

    # reduce feature space of joint features
    joint_features = reduce_feature_space(joint_features)
    joint_list = []
    for i in joint_features:
        print(str(i))
        joint_list.append(str(i))
    print('Joint features:', len(joint_features))

    # Combine the lists into a list of lists
    data = [joint_list, feature_list]

    # Save the data to a text file
    with open(filename, 'w') as file:
        for sublist in data:
            file.write(' '.join(sublist) + '\n')

    return final_constraints

def load_dataset_mine_constraints(trace_file, label_file, min_sup):
    #trace_file: A list of trace (strings)
    #label_file: A list of labels corresponding to the trace (strings)
    #min_sup: The minimum support threshold
    traces = []
    label_list = []
    labels = set()
    constraints_per_label = {}

    for trace, label in zip(trace_file, label_file):
        traces.append(trace)
        label_list.append(label)
        labels.add(label)

    traces_per_label = {}
    annotated_traces_per_label = {}

    for la, current_label in enumerate(sorted(labels)):
        if VERBOSE:
            print('\nCurrent label: ', current_label)
        final_traces = []
        
        ##### This is step 1 #########
        activity_count = {} #the activity count dictionary is used to count the occurences of each activity in the traces of the current label
        for label, trace in zip(label_list, traces):
            if label == current_label:
                trace = trace.replace('\n', '')
                acts = trace.split()
                for act in acts:
                    if act not in activity_count.keys():
                        activity_count[act] = 0
                    activity_count[act] += 1
                final_traces.append(acts)
        traces_per_label[label] = final_traces

        non_redundant_activities = set()
        for act, count in activity_count.items():
            if count >= len(final_traces) * min_sup: # the non-redundant activities are kept if they are better than the minimal support
                non_redundant_activities.add(act)
        if VERBOSE:
            print('#non-redundant activities: ', len(non_redundant_activities))
        non_redundant_activities = sorted(non_redundant_activities)

        ###############
        ####### Step 2: Mining the constraints###########""
        print('step 2: mine the constraints')
        constraints_for_label, annotated_traces = mine_constraints(final_traces, non_redundant_activities, current_label, min_sup)
        
        #for i in annotated_traces:
        #    print('constraints', i.print_constraints())
        
        annotated_traces_per_label[current_label] = annotated_traces
        
        constraints_per_label[current_label] = constraints_for_label
    return constraints_per_label, annotated_traces_per_label

def load_dataset_check_constraints(trace_file, label_file, activities):
    traces = []
    label_list = []
    labels = set()
    constraints_per_label = {}
    for trace, label in zip(trace_file, label_file):
        traces.append(trace)
        label_list.append(label.replace('\n', ''))
        labels.add(label.replace('\n', ''))

    annotated_traces_per_label = {}
    for la, current_label in enumerate(labels):
        if VERBOSE:
            print('\nCurrent label: ', current_label)
        final_traces = []
        activity_count = {}
        for label, trace in zip(label_list, traces):
            if label == current_label:
                acts = trace.split()
                for act in acts:
                    if act not in activity_count.keys():
                        activity_count[act] = 0
                    activity_count[act] += 1
                final_traces.append(acts)
        constraints_for_label, annotated_traces = mine_constraints(final_traces, activities,
                                                                              current_label, 0)
    
        constraints_per_label[current_label] = constraints_for_label
        annotated_traces_per_label[current_label] = annotated_traces
        
    return constraints_per_label, annotated_traces_per_label

def mine_constraints(traces, non_redundant_activities, label, min_sup):
    if VERBOSE:
        print('number of traces', len(traces))
        print('label',  label)

    constraint_count = {}
    actual_traces = 0
    annotated_traces = []
    
    print('Non redundant:', non_redundant_activities)

    for t, trace in enumerate(traces):
        #print(trace)
        #if t % 100 == 0 and t > 0:
        #    print('Doing trace',t)
        constraints = set()           
        actual_traces += 1
        miner = ConstraintMining(trace, label, non_redundant_activities)
        constraints = miner.FindConstraints()
        for constraint in constraints:
            if constraint not in constraint_count.keys():
                    constraint_count[constraint] = 0
            constraint_count[constraint] += 1
        annotated_traces.append(Annotated_trace(trace, constraints, label))

        #for i in annotated_traces:
        #    print(i.print_constraints())
        
    
    #for i in constraint_count:
    #    print('constraints',i, constraint_count[i])
    #print(len(traces))

    if VERBOSE:
        print('#constraints prior removal: ', len(constraint_count))

    to_remove = set()
    for constraint, count in constraint_count.items():
        #print('Actual traces:', actual_traces)
        if count < (actual_traces * min_sup):
            to_remove.add(constraint)
    for tr in to_remove:
        del constraint_count[tr]

    #for i in constraint_count:
    #    print('constraints',i, constraint_count[i])

    return constraint_count.keys(), annotated_traces

def iBCM_verify(filename, traces, labels, constraints):
    # this function is used for labeling the traces based on a set of constraints
    print('\nRunning iBCM labelling ', filename, ' for (', len(constraints), ' constraints)')
    
    used_activities = set()
    for constraint in constraints:
        used_activities.add(constraint.a)   # adding activity A
        used_activities.add(constraint.b)   # adding activity B

    print('Used activities:', len(used_activities))
    print('Used constraints:', len(constraints))
    
    constraints_per_label, annotated_traces = load_dataset_check_constraints(traces, labels, used_activities)


    for label in annotated_traces.keys():
        for trace in annotated_traces[label]:
            trace = trace.constraints.intersection(constraints)

    output = open(filename, 'w')
    for constraint in constraints:
        output.write(str(constraint) + ',')
    output.write('label\n')
    
    for label in annotated_traces.keys():
        for trace in annotated_traces[label]:
            for constraint in constraints:
                if constraint in trace.constraints:
                    output.write('1,')
                else:
                    output.write('0,')
            output.write(label + '\n')
    output.close()
