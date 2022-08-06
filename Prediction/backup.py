#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 22:09:32 2022

@author: roy
"""

    def apply_rules_to_inputs(self):    
        
        rule_output_strength = np.empty([len(self.train_object.reduced_rules), 1])
        rule_output_strength.fill(np.NaN)
        rule_with_strength = np.hstack((rule_output_strength,self.train_object.reduced_rules))
        # for each value, get strength for each rule
        for rule_index , rule in enumerate(self.train_object.reduced_rules):
                                                                                        # out of range?
            rule_with_strength[rule_index][0] = self.individial_rule_output(self.past_point_firing_strengts, rule[0:self.train_object.p])
        
        working_rules = []
        # get the rules that have strengths
        for rrule in rule_with_strength:
            if rule[0] != "nan":
                working_rules.append(rule)
        working_rules = np.asarray(working_rules)
        firing_level_for_each_output = self.union_strength_of_same_antecedents(working_rules)
        
        # calculate the centroid of the united outputs
        centroid = self.generate_outputs_object(firing_level_for_each_output)
        return centroid


    
    def union_strength_of_same_antecedents(self, working_rules):
        
        grouped_output_antecedent_strength = pd.DataFrame(index=list(range(0, len(working_rules))), columns=list(range(1, 3)))
        
        grouped_output_antecedent_strength[1] = working_rules[:,0]
        grouped_output_antecedent_strength[2] = working_rules[:,-2]
                
        l1 = grouped_output_antecedent_strength.groupby([2]).max()
        return(list(zip(l1.index, l1[1])))  

    




    def apply_rules_to_inputs(self):    
        
                              
        rule_output_strength = np.empty([len(self.train_object.reduced_rules), 1])
        rule_output_strength.fill(np.NaN)
        # for each value, get strength for each rule
        for rule_index , rule in enumerate(self.train_object.reduced_rules):
                                                                                        # out of range?
            rule_output_strength[rule_index] = self.individial_rule_output(self.past_point_firing_strengts, rule[0:self.train_object.p])

        firing_level_for_each_output = self.union_strength_of_same_antecedents(rule_output_strength, self.train_object.reduced_rules[:,self.train_object.p])
        
        # calculate the centroid of the united outputs
        centroid = self.generate_outputs_object(firing_level_for_each_output)
                
        return (centroid)
        
    def union_strength_of_same_antecedents(self, list_of_antecedent_strength, output_antecedent_list):
        
        grouped_output_antecedent_strength = pd.DataFrame(index=list(range(0, len(output_antecedent_list))), columns=list(range(1, 3)))
        
        grouped_output_antecedent_strength[1] = list_of_antecedent_strength
        grouped_output_antecedent_strength[2] = output_antecedent_list
                
        l1 = grouped_output_antecedent_strength.groupby([2]).max()
        #print(l1)
        l1 = pd.DataFrame.dropna(l1)
        #print(list(zip(l1.index,l1[1])))
        return(list(zip(l1.index, l1[1])))   