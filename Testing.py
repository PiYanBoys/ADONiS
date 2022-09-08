

import pandas as pd
import numpy as np

from T1_set import T1_Gaussian, T1_Triangular, T1_RightShoulder, T1_LeftShoulder
from inter_union import inter_union
from T1_output import T1_Triangular_output, T1_RightShoulder_output, T1_LeftShoulder_output



class testing(object):
    
    def __init__(self,train_object,  test_data, noise_free_test):
         
        self.noise_free_test = noise_free_test
        self.test_data = np.hstack(test_data)
        self.train_object = train_object
        
    # firing strengths of each input to each antecedent     
    def generate_firing_strengths(self, advaced_NSFLSs, my_technique, nf_sigma, est_past_points):
        
        self.all_firing_strengts = np.empty([len(self.test_data), len(self.train_object.antecedents)])
        self.all_firing_strengts2 = np.empty([len(self.test_data), len(self.train_object.antecedents)])
        self.all_firing_strengts.fill(np.NaN)

        #Defining non-adaptive sigma values
        sigmas = [0.0, nf_sigma/10, nf_sigma/(10**0.5), nf_sigma]
    
        #Non-Adaptive manually adjusting the sigma value of the input MFs
        if(my_technique=="singleton"):
            sigma_of_input=sigmas[0]
        elif(my_technique=="sigma_20"):
            sigma_of_input=sigmas[1]
        elif(my_technique=="sigma_10"):
            sigma_of_input=sigmas[2]
        elif(my_technique=="sigma_0"):
            sigma_of_input=sigmas[3]
        
        
        for index, input_val in enumerate(self.test_data):

            #Using the proposed ADONiS to adapt sigma value of the input MFs
            if(my_technique=="ADONiS"):
                sigma_of_input=self.get_input_sigma_value(index,est_past_points)
            
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
            self.all_firing_strengts[index] = temp_firings   
       
    # adaptive sigmas for every antecedent's input
    # 9 is number of past points
    def get_input_sigma_value(self, index_no, est_past_points):
        if est_past_points>index_no:
            # 两数据集有重叠才需要-est_past_points*2，否则不用*2
            ttt=np.asarray(self.train_object.train_data[(len(self.train_object.train_data)-est_past_points-9+index_no+1):(len(self.train_object.train_data)-est_past_points)])

            return(self.noise_estimation(np.concatenate((ttt,self.test_data[0:index_no+1]))))
        else:
            return(self.noise_estimation(self.test_data[index_no+1-est_past_points:index_no+1]))    
        
        
    def apply_rules_to_inputs(self):    
        
        output_results = []
                      
        for input_index , i  in enumerate(self.all_firing_strengts):            
            rule_output_strength = np.empty([len(self.train_object.reduced_rules), 1])
            rule_output_strength.fill(np.NaN)
            # for each value, get strength for each rule
            for rule_index , rule in enumerate(self.train_object.reduced_rules):
                                                                                            # out of range?
                rule_output_strength[rule_index] = self.individial_rule_output(self.all_firing_strengts[input_index:(input_index + self.train_object.p)], rule[0:self.train_object.p])

            firing_level_for_each_output = self.union_strength_of_same_antecedents(rule_output_strength, self.train_object.reduced_rules[:,self.train_object.p])
            
            # calculate the centroid of the united outputs
            centroid = self.generate_outputs_object(firing_level_for_each_output)
            output_results.append(centroid)
                
        
        SMAPE= self.get_smape(np.asarray(self.noise_free_test[self.train_object.p:]), np.asarray(output_results[:-self.train_object.p]))
        MSE = self.get_MSE(self.noise_free_test[self.train_object.p:], output_results[:-self.train_object.p]) 
        
        return (SMAPE)
        
    def union_strength_of_same_antecedents(self, list_of_antecedent_strength, output_antecedent_list):
        
        grouped_output_antecedent_strength = pd.DataFrame(index=list(range(0, len(output_antecedent_list))), columns=list(range(1, 3)))
        
        grouped_output_antecedent_strength[1] = list_of_antecedent_strength
        grouped_output_antecedent_strength[2] = output_antecedent_list
                
        l1 = grouped_output_antecedent_strength.groupby([2]).max()
        #print(l1)
        l1 = pd.DataFrame.dropna(l1)
        #print(list(zip(l1.index,l1[1])))
        return(list(zip(l1.index, l1[1])))   
    
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
        for  index_of_ant, fs in pairs_of_strength_antecdnt:
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
        #print(self.train_object.antecedents.keys())
        try:
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
