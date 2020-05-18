"""
python plot_results.py results/mar26_1best_1random_debug.txt 19 
"""

from matplotlib import pyplot as plt
from pylab import *
import numpy as np
import math
import sys

def extract_c_r_list(iteration=0, match_str = "Train Point Scores", fname = None): 
    c_r_list = []
    with open (fname, 'r') as f_rd:
        data = f_rd.readlines()
    found = False
    for line in data:
        if 'Outerloop: Iteration' in line:
            if str(iteration) in line:
                found = True
        if found == True and match_str in line:
            if match_str == 'train_sorted_c_r_list':
                c_r_list_str = line.strip(']\n').split('[')[-1] 
                from ast import literal_eval as createTuple
                c_r_list = list(createTuple(c_r_list_str))
            else:
                c = float(line.strip(')\n').split(' ')[-2])
                r = float(line.strip(')\n').split(' ')[-1])
                c_r_list.append((c, r))

        #Return if the current iteration is done
        if 'Train and Test mse' in line and found==True:
            return c_r_list


#******************RANK LOSS PLOT**************************
def maximize_window():
    #Maximize figure window
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()

    #Increase the size of the figure before saving it
    figure = plt.gcf() # get current figure
    figure.set_size_inches(14, 8)

def plot_rank_loss():
    inputs = range(-5, 6, 1)
    rank_loss1 = []
    rank_loss2 = []
    for diff in inputs:
        rank_loss1.append(-1*diff + np.log(1 + np.exp(diff)))
        rank_loss2.append(0*diff + np.log(1 + np.exp(diff)))
    
    plt.plot(inputs, rank_loss1, label='Rank A > Rank B')
    plt.plot(inputs, rank_loss2, label='Rank A < Rank B')
    xlabel('(Score A - Score B) (Higher score means higher rank)',fontsize=12)
    ylabel('Rank Loss',fontsize=18)
    legend(loc='top center', prop={'size':12}, ncol=10)
    
    #plt.show()
    plt.savefig("rankloss.png")

def normalize(data, normalize_flag, mean=None, std=None): 
    if normalize_flag == True:
        mean = np.mean(data)
        std = np.std(data)
        #print ('Mean and Variance', mean, std)
    data = [(d -mean)/std for d in data]
    return (data, mean, std)

def extract_x_y(sorted_c_r_list, invert=False, normalize_flag=False, mean=None, std=None):
    x = [c for (c,r) in sorted_c_r_list]
    if invert==True:
        y = [1.0-r for (c,r) in sorted_c_r_list]
    else:
        y = [r for (c,r) in sorted_c_r_list]
    y, mean, std = normalize(y, normalize_flag, mean, std)
    return (x, y, mean, std)


def plot_data(x, y, y_error, color_list, skip=False, label=None):
    #my_xticks = ['ReLu', 'Swish', 'tanh', 'max(x, sigma(x))'] 
    #plt.xticks(x, my_xticks)
    if y_error is None: 
        y_error = [0.01]*len(y)
    for i in range(len(x)):
        #print (x[i], y[i])
        color = color_list[i]
        if x[i] < 1.5 and x[i] > 0.41: 
            #color = 'c'
            if skip == True:
                continue
        plt.errorbar(x[i], y[i], y_error[i], marker='D', markerfacecolor=color, markeredgecolor=color_list[i], label= label if i==0 else '')
    #ylim(0,0.7)
    return

def remove_train_from_test(train_c_r_list, test_c_r_list):
    del_list = []
    for (c1, r1) in train_c_r_list: 
        for i, (c2, r2) in enumerate(test_c_r_list):
            if round(c1, 8) == round(c2, 8):  #rounding to 8th decimal digit
                if r1 != r2: 
                    print ('Warning: train c is the same but the r is different between train_c_r_list and test_c_r_list');
                    print (c1, r1, c2, r2)
                del_list.append(i)
    test_notrain_c_r_list = []
    for i, (c2, r2) in enumerate(test_c_r_list):
        if i not in del_list:
            test_notrain_c_r_list.append((c2, r2))
    return test_notrain_c_r_list

def plot_comparison(outerloop_iter, logFullPath, figDirFullPath):
    import pickle
    import sys
    import glob
    
    file_list = glob.glob(logFullPath+'*')
    #plt.ylim(-8.0, 2.2)

    #Full Ground Truth
    with open("mar8_search_slope_full.pkl", 'rb') as f_rd: 
         full_sorted_c_r_list = pickle.load(f_rd)
    x, y, _, _ = extract_x_y(full_sorted_c_r_list, invert=False, normalize_flag=True)
    color_list = ['b' for i in range(len(x))]
    plot_data(x, y, None, color_list, label='Ground-truth performance of 2-layer, 100 wide MNIST network')
    
    ##Partial Ground Truth (Points from the ground truth that are seen during hypertraining)
    #train_sorted_c_r_list = extract_c_r_list(iteration=outerloop_iter, match_str = "train_sorted_c_r_list", fname = file_list[0])
    #x, y, _, _ = extract_x_y(train_sorted_c_r_list, invert=False, normalize_flag=True)
    #color_list = ['y' for i in range(len(x))]
    #plot_data(x, y, None, color_list, label='Partial Ground Truth Performance')

    #Plot training data data (Starts from 0)
    #from plot_data import train_sorted_c_r_list
    test_y_list = []
    train_y_list = []
    for fname in file_list[:]:
        train_sorted_c_r_list = extract_c_r_list(iteration=outerloop_iter, match_str = "Train Point Scores", fname = fname)
        train_x, train_y, mean, std = extract_x_y(train_sorted_c_r_list, invert=True, normalize_flag=True)
        train_y_list.append(train_y)

        #Plot test data data (Use mean and std values computed during training)
        test_c_r_list = extract_c_r_list(iteration=outerloop_iter, match_str = "Test Point Scores", fname = fname)
        test_notrain_c_r_list = remove_train_from_test(train_sorted_c_r_list, test_c_r_list) 
        test_x, test_y, _, _ = extract_x_y(test_notrain_c_r_list, invert=True, normalize_flag=False, mean=mean, std=std)
        test_y_list.append(test_y)
    
    train_y_avg = np.mean(train_y_list, axis=0)
    train_y_std = np.std(train_y_list, axis=0)
    train_y_std_err = train_y_std/(math.sqrt(len(train_y_std))) #Std error is computed by dividing the std-dev by square-root(N)
    color_list = ['g' for i in range(len(train_x))]
    plot_data(train_x, train_y_avg, train_y_std_err, color_list, skip = False, label=None)
     
    test_y_avg = np.mean(test_y_list, axis=0)
    test_y_std = np.std(test_y_list, axis=0)
    test_y_std_err = test_y_std/(math.sqrt(len(test_y_std)))  #Std error is computed by dividing the std-dev by square-root(N)
    color_list = ['g' for i in range(len(test_x))]
    plot_data(test_x, test_y_avg, test_y_std_err, color_list, skip = True, label='Synthetic Petri Dish Model Predicted Performance')

    with open("model_predicted_train.pkl", 'rb') as f_rd: 
         model_train_c_r_list = pickle.load(f_rd)
    x, y, mean, std = extract_x_y(model_train_c_r_list, invert=False, normalize_flag=True)
    color_list = ['r' for i in range(len(x))]
    plot_data(x, y, None, color_list, skip = False, label=None)

    with open("model_predicted_test.pkl", 'rb') as f_rd: 
         model_test_c_r_list = pickle.load(f_rd)
    x, y, _, _ = extract_x_y(model_test_c_r_list, invert=False, normalize_flag=False, mean=mean, std=std)
    color_list = ['r' for i in range(len(x))]
    plot_data(x, y, None, color_list, skip = False, label='Neural Network Model Predicted Performance')

    maximize_window()
 
    xlabel('Sigmoid Slope Value',fontsize=20)
    ylabel('Normalized Accuracy',fontsize=20)
    #title('Iteration '+str(outerloop_iter)+' in the Petri-dish') 
    #title('Normalized Accuracy v/s. Sigmoid Slope') 
    legend(loc='top', prop={'size':17.5})
    xticks(fontsize=18)
    yticks(fontsize=18)
    axvspan(0.41, 1.5, facecolor='lightsteelblue', alpha=0.5)
    plt.savefig(figDirFullPath+'/'+str(outerloop_iter)+".png", format='png', dpi=500, bbox_inches='tight')
    #plt.show()
    os.system('open '+figDirFullPath+'/'+str(outerloop_iter)+".png") 
    close()
    return

if __name__=="__main__":
    
    logFullPath = str(sys.argv[1])  #results/mar26.txt
    max_outerloop_iter = int(sys.argv[2])  #0/1/2...
    
    figDirName = logFullPath.strip().split('/')[1].split('.')[0] #Extract mar26 from results/mar26.txt 
    figDirFullPath = 'figures/'+figDirName 
    
    #Create Directory if it does not exist
    import os
    if not os.path.exists(figDirFullPath):
        os.makedirs(figDirFullPath)
    
    for outerloop_iter in range(max_outerloop_iter):
        plot_comparison(outerloop_iter, logFullPath, figDirFullPath)
