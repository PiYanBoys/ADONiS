import numpy as np
from operator import mul
from T1_set import *
from functools import reduce


class wang_mendel(object):
    
    def __init__(self, train_data, antecedent_number, past_points):

        self.train_data=train_data
        self.antecedents=self.generate_antecedents(train_data,antecedent_number)
        self._p=past_points
        self.__reduced_rules=self.rule_matrix_generating()
    #number of antecdents
    def generate_antecedents(self,train_data,antecedent_number):
            
            max_number = max(train_data)
            min_number = min(train_data)
            
            antecedents = {}
            step = ((max_number - min_number) / (antecedent_number - 1)) / 4.0      
            for i in range(1, antecedent_number + 1):
                
                mean = min_number + (i - 1) * step * 4.0
                if(i == 1):
                     antecedents[i] = T1_LeftShoulder(mean, step , 500)
                elif(i == antecedent_number):
                    antecedents[i] = T1_RightShoulder(mean, step , 500)
                else:
                    antecedents[i] = T1_Triangular(mean, step , 500)
            #print(antecedents[1].interval[0])
            #print(antecedents[7].interval[1])
            #print(min_number)
            #print(max_number)
            return (antecedents)
    
    
    def rule_matrix_generating(self):
        #[ value of train data, degree of mf, index of mf ]        
        x_val_with_mf= self.assign_points()
        
        all_rule_matrix = np.zeros([len(self.train_data) - self.p, (self.p + 2)])
        
        for i in range(0, len(self.train_data) - self.p):
            all_rule_matrix[i][0:self.p+1]=np.hstack(x_val_with_mf[i:(i+self.p+1),2:3])
            
            #multiplication of degrees
            rule_degree = reduce(mul, x_val_with_mf[i:(i+self.p+1),1:2])
            
            all_rule_matrix[i][self.p+1:self.p+2]=rule_degree
            #second to last is consequent, the last is rule degree, the rest are antecedents
        return(self.rule_reduction(all_rule_matrix))
        #return (all_rule_matrix)
    
    # assign the best matching mf and degree to every value of the dataset
    def assign_points(self):
        x_val_with_mf = np.empty([len(self.train_data), (3)])    
        
        for index,x in enumerate(self.train_data):
            x_val_with_mf[index][0]=x
            x_val_with_mf[index][1:3]=self.get_antIndex_and_maxDegree(x)
            
        return(x_val_with_mf) 
       
    def get_antIndex_and_maxDegree(self,x):
        
        max_degree=0.0
        
        for i in self.antecedents:
            degree=self.antecedents[i].get_degree(x)
            if(degree>max_degree):
                max_degree=degree
                antIndex=i
        
    
        if(max_degree==0.0):
            raise ValueError( "There is no max degree")
        else:
            return ((max_degree,antIndex))    
    
    
    def rule_reduction(self,all_rule_matrix):

        for i in range(0, len(all_rule_matrix)):
            temp_rule_1 = all_rule_matrix[i]
            if not np.isnan(temp_rule_1).any() :
                for t in range(i + 1, len(all_rule_matrix)):
                    temp_rule_2 = all_rule_matrix[t]
                    # check antecedent equality
                    if np.array_equal(temp_rule_1[0:(self.p)], temp_rule_2[0:(self.p)]):
                        # check degree and assign greater one
                        if(temp_rule_2[self.p + 1] > temp_rule_1[self.p + 1]):
                            # the rule,with lower degree, is replaced by the higher degree one
                            all_rule_matrix[i] = all_rule_matrix[t]
                        all_rule_matrix[t] = np.nan
                       
        return(all_rule_matrix[~np.isnan(all_rule_matrix).any(axis=1)])
        
          
            
                       
    @property
    def mf_interval_matrix(self):
        return self.__mf_interval_matrix
    
    @property
    def reduced_rules(self):
        return self.__reduced_rules
    
    @property
    def p(self):
        return self._p
    
    @property
    def shape(self):
        return self._shape

  

#np.savetxt(" .csv",self.reduced_rules,delimiter=",")