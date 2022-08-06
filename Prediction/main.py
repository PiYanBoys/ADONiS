

import pandas as pd
import numpy as np
import random
from scipy import stats
import matplotlib.pyplot as plt
import sys

from train_wm import wang_mendel
from Testing import testing
    

def generate_noise_with_db(db,nf_sigma):
    sigma_noise=nf_sigma/float(10**(db/20.0))
    return(stats.norm.ppf(random.random(), loc=0, scale=sigma_noise))

def get_noisy_set(test_set_name,current_series,nf_sigma):
    
    if(test_set_name=="Stable_Noise"):
        db_variance=0
        list_of_db=[10,10,10,10,10,10,10,10,10]
    elif(test_set_name=="Mixed_Stable_Noise"):
        db_variance=0
        list_of_db=[20,15,10,5,0,5,10,15,20]
    elif(test_set_name=="Variable_Noise"):
        db_variance=10
        list_of_db=[20,15,10,5,0,5,10,15,20]
            

    noisy_list=np.zeros(len(current_series))
    noise_leves_in_dBs=[]
    
    t=0
    for index, i in enumerate(list_of_db):
        previous_t=t
        if(index in [0,4,8]):
            length=(len(current_series)-30)/3
            t=t+int(length)
        else:
            length=5
            t=t+int(length)
        #print(length)
        #print(previous_t)
        #print(t)
        #print()
        for current_t in range(previous_t,t): 
            db=random.uniform(-db_variance,db_variance)+i
            noise_leves_in_dBs.append(db)
            noisy_list[current_t]=(current_series[current_t] + generate_noise_with_db(db,nf_sigma))
            
    #print(pd.Series((v for v in noisy_list)))
    return(pd.Series((v for v in noisy_list)))

  
def plot_the_results(main_results,my_experiment):
   
    plt.figure(figsize=(10,5))
    plt.rcParams.update({'font.size': 20})
    plt.ylabel("sMAPE")
    plt.gca().yaxis.grid(True)  
        
    if (my_experiment==1.1 or my_experiment==1.2):
        
        
        plt.bar([1,8,15],[np.average([row[0][0] for row in main_results]),\
                          np.average([row[1][0] for row in main_results]),\
                              np.average([row[2][0] for row in main_results])],\
                edgecolor='black', hatch="//",label="Adaptive")
        plt.bar([2,9,16],[np.average([row[0][1] for row in main_results]),\
                          np.average([row[1][1] for row in main_results]),\
                              np.average([row[2][1] for row in main_results])], \
                edgecolor='black', hatch="-",label="Singleton")
        plt.bar([3,10,17],[np.average([row[0][2] for row in main_results]),\
                           np.average([row[1][2] for row in main_results]),\
                               np.average([row[2][2] for row in main_results])], \
                edgecolor='black', hatch="o",label=r"$\sigma_{20}$")
        plt.bar([4,11,18],[np.average([row[0][3] for row in main_results]),\
                           np.average([row[1][3] for row in main_results]),\
                               np.average([row[2][3] for row in main_results])], \
                edgecolor='black', hatch="x",label=r"$\sigma_{10}$")
        plt.bar([5,12,19],[np.average([row[0][4] for row in main_results]),\
                           np.average([row[1][4] for row in main_results]),\
                               np.average([row[2][4] for row in main_results])], \
                edgecolor='black', hatch="*",label=r"$\sigma_{0}$")
     
    if (my_experiment==2.1 or my_experiment==2.2):
        
        plt.bar([1,8,15],[np.average([row[0][0] for row in main_results]),\
                          np.average([row[1][0] for row in main_results]),\
                              np.average([row[2][0] for row in main_results])], \
                edgecolor='black', hatch="//",label="Adaptive")
        plt.bar([1,8,15],[0,0,0], edgecolor='black', hatch="\\")
        plt.bar([2,9,16],[0,0,0], edgecolor='black', hatch=".")
        plt.bar([3,10,17],[0,0,0], edgecolor='black', hatch="--")
        plt.bar([3,10,17],[0,0,0], edgecolor='black', hatch="--")
        plt.bar([3,10,17],[0,0,0], edgecolor='black', hatch="--")
        plt.bar([2,9,16],[np.average([row[0][1] for row in main_results]),\
                          np.average([row[1][1] for row in main_results]),\
                              np.average([row[2][1] for row in main_results])], \
                edgecolor='black', hatch="O",label="Cen-NS")
        plt.bar([3,10,17],[np.average([row[0][2] for row in main_results]),\
                           np.average([row[1][2] for row in main_results]),\
                               np.average([row[2][2] for row in main_results])], \
                edgecolor='black', hatch="+",label="Sim-NS")
        plt.bar([4,11,18],[np.average([row[0][3] for row in main_results]),\
                           np.average([row[1][3] for row in main_results]),\
                               np.average([row[2][3] for row in main_results])], \
                edgecolor='black', hatch="||",label="Sub-NS")

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,  ncol=5, mode="expand", borderaxespad=0.,fontsize=15)
    plt.xticks([3,10,17], ["Stable Noise","Mixed Stable Noise","Variable Noise"])
    plt.savefig("results/Figure"+str(my_experiment)+".png")
    print("\nCompleted! Please see the generated Figure"+str(my_experiment))

def print_loading_percentage(count_of_iteration, total_iteration):
    current_percentage=(count_of_iteration*100)/float(total_iteration)

    sys.stdout.write("\r%d%%" % current_percentage)
    sys.stdout.write(" ||")
    for i in range(100):
        if i<=current_percentage:
            sys.stdout.write("#")
        else:
            sys.stdout.write(" ")
    sys.stdout.flush()
    sys.stdout.write("||")            
    
    


def main(antecedent_number=7, predict_past_points=9, noise_est_past_points=9, NSFLSs="standard", technique="ADONiS"):
    
    noise_free_train = pd.read_csv("timeser/lorenz_train.csv", header=None)[1]
    
        #Generating rules from noise free set
    train_obj = wang_mendel(noise_free_train, antecedent_number, predict_past_points)

    test_obj = testing(train_obj, noise_free_train) 
    
    test_obj.generate_firing_strengths(NSFLSs,technique,noise_est_past_points)
    
    result = test_obj.apply_rules_to_inputs()
    
    
    print("The next point will be around: ",result)
    
"""
                if(my_experiment==1.1 or my_experiment==1.2):
                    prediction_results[index_test_set,index_technique] = test_obj.apply_rules_to_inputs()
                if(my_experiment==2.1 or my_experiment==2.2):
                    prediction_results[index_test_set,index_advaced_NSFLSs] = test_obj.apply_rules_to_inputs()
                  """

                    
    #main_results.append(prediction_results)
        
    #Complete the loading bar
    #print_loading_percentage(count+1, repeats*len(test_sets)*len(NSFLSs)*len(techniques))  
    
    #plot_the_results(main_results,my_experiment)

"""
if __name__ == "__main__":
    print(" ")
    main(str(sys.argv[1]),sys.argv[2],sys.argv[3])
    """