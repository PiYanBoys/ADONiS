

import pandas as pd
import numpy as np

from T1_set import T1_Gaussian, T1_Triangular, T1_RightShoulder, T1_LeftShoulder
from inter_union import inter_union
from T1_output import T1_Triangular_output, T1_RightShoulder_output, T1_LeftShoulder_output



class testing(object):
    
    def __init__(self,train_object,  test_data):
         
        self.test_data = np.hstack(test_data)
        self.train_object = train_object
        
    # firing strengths of each input to each antecedent     
    def generate_firing_strengths(self, advaced_NSFLSs, my_technique, noise_est_past_points):
        
        self.past_point_firing_strengts = np.empty([self.train_object.p, len(self.train_object.antecedents)])
        self.past_point_firing_strengts.fill(np.NaN)

        #Defining non-adaptive sigma values
        sigmas=[0.0,0.0286982683,0.0907518928,0.2869826834]

        #Non-Adaptive manually adjusting the sigma value of the input MFs
        if(my_technique=="singleton"):
            sigma_of_input=sigmas[0]
        elif(my_technique=="sigma_20"):
            sigma_of_input=sigmas[1]
        elif(my_technique=="sigma_10"):
            sigma_of_input=sigmas[2]
        elif(my_technique=="sigma_0"):
            sigma_of_input=sigmas[3]
        
        for index, input_val in enumerate(self.test_data[-self.train_object.p:]):

            #Using the proposed ADONiS to adapt sigma value of the input MFs
            if(my_technique=="ADONiS"):
                sigma_of_input=self.get_input_sigma_value(index,noise_est_past_points)
            print("input_val",input_val)
            #create input objects
            input_obj = T1_Gaussian(input_val, sigma_of_input, 500)   
                      
            temp_firings = []

            for i in self.train_object.antecedents:
                if(my_technique=="singleton"):
                    fs = self.train_object.antecedents[i].get_degree(input_val)
                else:
                    inter_un_obj = inter_union(self.train_object.antecedents[i], input_obj, 500)
                    fs = inter_un_obj.return_FSSSS(advaced_NSFLSs)  
                temp_firings.append(fs)
            self.past_point_firing_strengts[index] = temp_firings   
    # adaptive sigmas for every antecedent's input
    # 9 is number of past points
    def get_input_sigma_value(self, index_no, noise_est_past_points):
        return(self.noise_estimation(self.test_data[ len(self.test_data)-self.train_object.p + index_no + 1 - noise_est_past_points : \
                                                     len(self.test_data)-self.train_object.p + index_no + 1]))    
   
            
    def apply_rules_to_inputs(self):    
        
        rule_with_strength = np.empty([len(self.train_object.reduced_rules), 1])
        rule_with_strength.fill(np.NaN)
        rule_with_strength = np.hstack((rule_with_strength,self.train_object.reduced_rules))
        temp = np.empty([len(self.train_object.reduced_rules),1])
        for i in range(len(rule_with_strength)):
            temp[i][0] = i
        rule_with_strength = np.hstack((temp,rule_with_strength))
        # for each value, get strength for each rule
        for rule_index , rule in enumerate(self.train_object.reduced_rules):
                                                                                        # out of range?
            rule_with_strength[rule_index][1] = self.individial_rule_output(self.past_point_firing_strengts, rule[0:self.train_object.p])
            # rule_with_strength like [ index , strength , ...rule... ]
        # firing_level_for_each_output like [ antecedent_index , strength , rule_index ]
        firing_level_for_each_output = self.union_strength_of_same_antecedents(rule_with_strength)
        
        # for printing working rules
        working_rules_matrix = np.empty([ len(firing_level_for_each_output) , self.train_object.p + 3 ])
        # working_rules_matrix like [ strength , ...rule... ]
        for index , every_antecedent in enumerate(firing_level_for_each_output):
            working_rules_matrix[index][0] = every_antecedent[1]
            working_rules_matrix[index][1:] = self.train_object.reduced_rules[int(every_antecedent[2])]
            print("Rule" , index + 1 , end = "")
            for index,i in enumerate(working_rules_matrix[index]):
                if index == 0:
                    print("strength:",i)
                elif index <= self.train_object.p + 1:
                    print(int(i),end=" , ")
                else :
                    print(i)
                    
        # calculate the centroid of the united outputs
        centroid = self.generate_outputs_object(firing_level_for_each_output)
        return centroid

    def union_strength_of_same_antecedents(self, rule_with_strength):
        
        grouped_output_antecedent_strength = pd.DataFrame(index=list(range(0, len(rule_with_strength))), columns=list(range(1, 4)))
        
        grouped_output_antecedent_strength[1] = rule_with_strength[:,1] # strength
        grouped_output_antecedent_strength[2] = rule_with_strength[:,-2] # output antecedent
        grouped_output_antecedent_strength[3] = rule_with_strength[:,0] # rule index
                
        l1 = grouped_output_antecedent_strength.groupby([2]).max()
        l1 = pd.DataFrame.dropna(l1)
        print(l1)
        return(list(zip(l1.index, l1[1], l1[3])))        
            
            

    def get_smape(self,A, F):
        return 100/float(len(A)) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))
        
    def get_MSE(self, real_values_list, predicted_value_list):
        
        return(np.square(np.subtract(real_values_list, predicted_value_list)).mean())

    # AND, get the minimun degree
    def individial_rule_output(self, inputs, rule):

        firing_level_of_pairs = 1
        
        for i in range(0, len(inputs)):
            temp_firing = inputs[i][int(rule[i]) - 1]
            
            if(temp_firing == 0):
                firing_level_of_pairs = "nan"
                break
                    
            # minimum is implemented
            if(temp_firing < firing_level_of_pairs):
                firing_level_of_pairs = temp_firing
            
        return firing_level_of_pairs
            
    def generate_outputs_object(self, pairs_of_strength_antecdnt):
             
        outputs = {}
        for  index_of_ant, fs, irrelevance in pairs_of_strength_antecdnt:
            if(isinstance(self.train_object.antecedents[index_of_ant], T1_Triangular)):
                outputs[index_of_ant] = T1_Triangular_output(fs, self.train_object.antecedents[index_of_ant].interval)
            if(isinstance(self.train_object.antecedents[index_of_ant], T1_RightShoulder)):
                outputs[index_of_ant] = T1_RightShoulder_output(fs, self.train_object.antecedents[index_of_ant].interval)
            if(isinstance(self.train_object.antecedents[index_of_ant], T1_LeftShoulder)):
                outputs[index_of_ant] = T1_LeftShoulder_output(fs, self.train_object.antecedents[index_of_ant].interval)
                #print(type())
        if(len(outputs) == 0):
            return 0
        
        degree = []
        try:
            # here question: interval
            disc_of_all = np.linspace(self.train_object.antecedents[list(self.train_object.antecedents.keys())[0]].interval[0],\
                                  self.train_object.antecedents[list(self.train_object.antecedents.keys())[-1]].interval[1],\
                                      int((500 / 2.0) * (len(self.train_object.antecedents) + 1)))
        except:
            print("error in generate outputs object")

        for x in disc_of_all:
            max_degree = 0.0
            for i in outputs:
                if max_degree < outputs[i].get_degree(x):
                    max_degree = outputs[i].get_degree(x)
            degree.append(max_degree) 
        
        numerator = np.dot(disc_of_all , degree)
        denominator = sum(degree)
        if denominator != 0:
            return(numerator / float(denominator))
        else:
            return (0.0)
                           

        
    def noise_estimation(self,testList):
        diff_list = []
        for i in range(len(testList)-1):
            diff_list.append((testList[i+1] - testList[i])/(2**0.5))
        #for index, i in enumerate(testList[:-1]):
            #diff_list.append((testList[index + 1] - i) / float(np.sqrt(2)))
        
        return(np.std(diff_list))     
