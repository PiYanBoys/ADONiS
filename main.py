

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

def get_noisy_set(test_set_name,current_series,nf_sigma,db_level):
    
    if(test_set_name=="Stable_Noise"):
        db_variance=0
        list_of_db=[db_level,db_level,db_level,db_level,db_level,db_level,db_level,db_level,db_level]
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

  
def plot_the_results(main_results,my_experiment,filename):
   
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
    plt.xticks([3,10,17], ["Steady","Blended Steady","Blended Random"])
    plt.savefig("results/Figure"+filename+".png")
    print("\nCompleted! Please see the generated Figure"+filename)

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
    
    


def main(my_experiment=1.1, repeats=1, antecedent_number=7, predict_past_points=9, est_past_points=9, db_level=10):
    
    my_experiment=float(my_experiment)

    repeats=int(repeats)
    filename = str(my_experiment) + "_" + str(antecedent_number) + "_" + str(predict_past_points) + "_" + str(est_past_points) + "_" + str(db_level)
    
    if(my_experiment not in [1.1,1.2,2.1,2.2]):
         raise ValueError('The experiment number can only be 1.1 , 1.2 , 2.1 or 2.2 and it must be specified in the run file.')
        
    if (type(repeats) is not int):
        raise ValueError('The repeat of experiment should be an integer and it must be specified in the run file.')

    
        
    if(my_experiment==1.1 or my_experiment==1.2):
        #See Supp. Materials
        NSFLSs=["standard"]
        techniques=["ADONiS", "singleton","sigma_20","sigma_10","sigma_0"]
              
    if(my_experiment==2.1 or my_experiment==2.2):
        #See Supp. Materials
        NSFLSs=["standard", "cen_NS", "sim_NS", "sub_NS"]
        techniques=["ADONiS"]
    
    #See Fig. 4 
    test_sets=["Stable_Noise","Mixed_Stable_Noise","Variable_Noise"] 
    
    #Read the noise_free time series
    noise_free_train = pd.read_csv("data/Lorenz/noise_free_train.csv", header=None)[0]
    
    #Read the noise free test set
    noise_free_test = pd.read_csv("data/Lorenz/noise_free_test.csv", header=None)[0]
    
    nf_sigma = pd.Series.std(noise_free_train)
    #noise_free_train = pd.read_csv("timeser/lorenz_train.csv", header=None)[1]
    
    #Read the noise free test set
    #noise_free_test = pd.read_csv("timeser/lorenz_test.csv", header=None)[1]
    #Noise free training
    if(my_experiment==1.1 or my_experiment==2.1):
        #Generating rules from noise free set
        train_obj = wang_mendel(noise_free_train, antecedent_number, predict_past_points)
    
       
        
    main_results=[]
    count=-1
    
    for repeat in range(repeats):
        prediction_results=np.zeros([3,5])
        
        for index_test_set, test_set_name in enumerate(test_sets):
            
            #Noisy Training
            if(my_experiment==1.2 or  my_experiment==2.2):
                                                               # why [:-1]        # noise free sigma? std?
                noisy_training_set=get_noisy_set(test_set_name, noise_free_train[:-1], nf_sigma, db_level)
                
                #Generating rules from noisy set
                train_obj = wang_mendel(noisy_training_set, antecedent_number, predict_past_points)    
                                                                        # why use train set sigma for test set?
            temp_noisy_set = get_noisy_set(test_set_name, noise_free_test, nf_sigma, db_level)
            
            for index_advaced_NSFLSs, advaced_NSFLSs in enumerate(NSFLSs):
                
                for index_technique, my_technique in enumerate(techniques):
                    
                    count=count+1
                    print_loading_percentage(count, repeats*len(test_sets)*len(NSFLSs)*len(techniques))                        
                        
                    test_obj = testing(train_obj, temp_noisy_set, noise_free_test) 
                    
                    test_obj.generate_firing_strengths(advaced_NSFLSs, my_technique, nf_sigma, est_past_points)
                    
                    if(my_experiment==1.1 or my_experiment==1.2):
                        prediction_results[index_test_set,index_technique] = test_obj.apply_rules_to_inputs()
                    if(my_experiment==2.1 or my_experiment==2.2):
                        prediction_results[index_test_set,index_advaced_NSFLSs] = test_obj.apply_rules_to_inputs()
                    

                    
        main_results.append(prediction_results)
        
    #Complete the loading bar
    print_loading_percentage(count+1, repeats*len(test_sets)*len(NSFLSs)*len(techniques))  
    
    plot_the_results(main_results,my_experiment,filename)


if __name__ == "__main__":
    print(" ")
    main(sys.argv[1])
    