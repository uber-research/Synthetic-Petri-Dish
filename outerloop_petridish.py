# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import helper
import bisect
import petridish
import numpy as np
import os
import autograd.numpy.random as npr

#0. Set petridish configuration, initial training data, weight initialization
def main(num_petridish_iter = 1, num_top_points = 2, perf_thresh = 0.97, layer_sizes= [2, 3, 1], L2_reg= 0.00001, param_scale = 0.15, batch_size = 20, num_epochs = 20, step_size = 0.01, hyper_iter = 600, hyper_step_size = 2.0, hyper_decay = 0.4, hyper_decay_after = 200, hyper_decay_every = 150, hyper_L2_reg = 0.00001, rank_loss_scaling_factor = 100.0, mse_tolerance = None, outputFname = '/tmp/results', train_sorted_c_r_list = None, full_sorted_c_r_list=None, fake_train_images = None, fake_train_labels = None, fake_valid_images = None, fake_valid_labels = None):
    
    seed = None 
    random_state=npr.RandomState(seed=seed) #seed is supplied from outerloop_petridish
    
    #Generate weight initialization
    #Following two lines will Generate new init file from Macbook with petridish (seed=0) for each new layer_size config 
    #init_params_list = petridish.generate_init_params(full_sorted_c_r_list, param_scale, layer_sizes, random_state)
    dir_path = os.path.dirname(os.path.realpath(__file__))+'/'
    init_param_fname = dir_path+'init_param_files/'+str(layer_sizes)+'_'+str(param_scale)+'.pkl'
    #helper.save_pickle(init_param_fname, init_params_list)
    #init_params_list = helper.load_pickle(init_param_fname)
    
    if fake_valid_images is None or fake_valid_labels is None:
        fake_valid_images = np.copy(fake_train_images)
        fake_valid_labels = np.copy(fake_train_labels)

    fake_train_images = random_state.randn(batch_size, layer_sizes[0])
    fake_train_labels = np.round(random_state.rand(batch_size, layer_sizes[-1])) #np.random.randint(2, size=(batch_size, layer_sizes[-1]))
    #fake_valid_images = random_state.randn(batch_sexp_numze, layer_sizes[0])
    #fake_valid_labels = np.random.randint(2, size=(batch_size, layer_sizes[-1]))
    fake_valid_images = np.copy(fake_train_images)
    fake_valid_labels = np.copy(fake_train_labels)

    #1. Warm up petridish with K number of points 
    #   a. call petridish with specific weight initialization, and other hyper-parameters 
    #   b. returns learnt training data
    
    #Sample K=4 random points (for now 6 fixed points)
    best_scores = [] #List of ground-truth values of the best predicted scores at each iteration. Start with the first K scores.
    for c, r in train_sorted_c_r_list:
        best_scores.append(r)  #for the warm-up, best scores have ground-truth values of the initial training points
    
    #Warming up Petridish
    success_flag = False
    iteration = 0
    lines = []  #Contains all the logs to be returned
    while iteration < num_petridish_iter:
    
        
        lines.append('Outerloop: Iteration '+ str(iteration)+'\n')
        lines.append('train_sorted_c_r_list ' + str(train_sorted_c_r_list)+'\n') 
    
        outputFname_iter = outputFname#+str(iteration)+'.txt'
        #if os.path.exists(outputFname_iter):
        #    os.remove(outputFname_iter) #Remove file if it exists so that we do not keep appending to it
        
        combined_init_params, mask_params, init_params = helper.get_init_param_list(1, train_sorted_c_r_list, param_scale, layer_sizes, None, random_state)
        #combined_init_params, mask_params = petridish.create_combined_init_params(init_params_list[0:len(train_sorted_c_r_list)], layer_sizes)
        
        #Train
        results = petridish.main(layer_sizes= layer_sizes, L2_reg= L2_reg, param_scale = param_scale, batch_size = batch_size, num_epochs = num_epochs, step_size = step_size, hyper_iter =hyper_iter, hyper_step_size = hyper_step_size, hyper_decay = hyper_decay, hyper_decay_after = hyper_decay_after, hyper_decay_every = hyper_decay_every, hyper_L2_reg = hyper_L2_reg, rank_loss_scaling_factor = rank_loss_scaling_factor, mse_tolerance = mse_tolerance, outputFname = outputFname_iter, sorted_c_r_list = train_sorted_c_r_list, init_params_list=None, combined_init_params = combined_init_params, mask_params = mask_params, random_state = random_state, hyper_train = True, fake_train_images = fake_train_images, fake_train_labels = fake_train_labels, fake_valid_images = fake_valid_images, fake_valid_labels = fake_valid_labels)
      
        #Extract trained data (This currently returns True if rank order correct)
        (train_mse, success_flag, train_score_list, learnt_images, learnt_train_labels, learnt_valid_images, learnt_valid_labels) = results
        for idx, (c, r) in enumerate(train_sorted_c_r_list):
            lines.append('Outerloop: Train Point Scores '+ str(c)+' ' + str(-train_score_list[idx])+'\n')
        
        ##If results did not converge, then check for intermediate convergence (first MSE convergence and then rank convergence)
        #if success_flag == False:
        #    extracted_learnt_images = helper.read_images_from_file(fname=outputFname_iter, num_images=len(fake_train_images), checkStr="MSE Loss converged")
        #    if extracted_learnt_images is None: #No MSE convergence
        #       extracted_learnt_images  = helper.read_images_from_file(fname=outputFname_iter, num_images=len(fake_train_images), checkStr="Rank order correct")
        #    else:
        #       print ('Outerloop: MSE converged')
        #if extracted_learnt_images is not None:
        #    learnt_images = extracted_learnt_images
        #    print ('Outerloop: Rank correct')
    
    #2. Run full grid-search (architecture search/auto-tune) with petridish
    #   a. Evaluate hundreds of points cheaply by running petridish in test/evaluation mode (no hyper-training)
    #   b. Take Y (say two) points from petridish (two best? or best and worst?)
    
        #Reduce the number of points over which we will do petridish search 
        #Exclude the points that we have already evaluated ground truth for. No need predicting those
        #For now, predict all for debugging purposes. But for selecting top predicted points, select from test_sorted_c_r_list (TODO)
        test_sorted_c_r_list = sorted(list(set(full_sorted_c_r_list)-set(train_sorted_c_r_list)), key = lambda tup: tup[1], reverse=True) #TODO: Change this approach of set
        #if len(test_sorted_c_r_list) != (len(full_sorted_c_r_list) - len(train_sorted_c_r_list)):
        #    print ("Outerloop: Error in Computing test_sorted_c_r_list"); import sys; sys.exit()
        #combined_init_params, mask_params = petridish.create_combined_init_params(init_params_list[0:len(test_sorted_c_r_list)], layer_sizes)
        combined_init_params, mask_params, _ = helper.get_init_param_list(1, test_sorted_c_r_list, param_scale, layer_sizes, init_params, random_state)
        
        #Test (Setting hyper_train=False and fake_train_images = learn_images)
        results = petridish.main(layer_sizes= layer_sizes, L2_reg= L2_reg, param_scale = param_scale, batch_size = batch_size, num_epochs = num_epochs, step_size = step_size, hyper_iter =hyper_iter, hyper_step_size = hyper_step_size, hyper_decay = hyper_decay, hyper_decay_after = hyper_decay_after, hyper_decay_every = hyper_decay_every, hyper_L2_reg = hyper_L2_reg, rank_loss_scaling_factor = rank_loss_scaling_factor, sorted_c_r_list = test_sorted_c_r_list, init_params_list=None, combined_init_params = combined_init_params, mask_params = mask_params, random_state = random_state, hyper_train = False, fake_train_images = learnt_images, fake_train_labels = fake_train_labels, fake_valid_images = learnt_valid_images, fake_valid_labels = learnt_valid_labels)
    
        #Extract Results 
        (test_mse, success_flag, test_score_list, learnt_images, learnt_train_labels, learnt_valid_images, learnt_valid_labels) = results 
        for idx, (c, r) in enumerate(test_sorted_c_r_list):
            lines.append('Outerloop: Test Point Scores '+ str(c)+' '+ str(-test_score_list[idx])+'\n')
        top_c_r_list = helper.extract_top_arch(test_score_list, test_sorted_c_r_list, K=num_top_points-1) #Extract top num_top_points-1 points with best predicted performance
        
        #Select one point randomly (not the best). This is to prevent the model from greedily selecting low-performance points in a local region
        #Ideally, this should be done with some random probability i.e initially, when the model is poor, do random selection
        #Later, when the model is more accurate, do only greedy selection
        reduced_c_r_list = [(c,r) for (c,r) in test_sorted_c_r_list if (c,r) not in top_c_r_list]
        randomly_selected_c_r_list = helper.randomly_select_K(reduced_c_r_list, perf_thresh, K=1) 

        top_c_r_list = top_c_r_list + randomly_selected_c_r_list
        
        lines.append('Train and Test mse '+ str(train_mse)+' '+ str(test_mse)+'\n')
    
    #3.   Evaluate ground truth for the points with best predicted score
        for idx, (c, r) in enumerate(top_c_r_list):
            if idx == num_top_points: 
                break
            best_scores.append(r) #Evaluate ground-truth for the best points and append to this list
    
    #4. Stopping condition - if we reached the best
        lines.append('Outerloop: Scores '+ str(best_scores)+'\n')
        for score in best_scores:
            if (score >= perf_thresh):
                lines.append('Outerloop: Found best solution'+str(best_scores)+'\n')
                return (lines, best_scores, train_mse, test_mse)
        
        lines.append('Adding Top predicted points '+str(top_c_r_list)+'\n') 
        #Continue - Add Y top (one or two) points to the train_sorted_c_r_list for further evaluation
        train_sorted_c_r_list = helper.add_new_arch_ground_truth(train_sorted_c_r_list, top_c_r_list)
        
    #5. Run petridish again with additional points (Restart or resume from learnt images/learn images+learnt weights?)
    
    #6. Go Back to 2.
        
        iteration = iteration + 1
    
    lines.append('Outerloop: NOT Found best solution'+str(best_scores)+'\n')
    return (lines, best_scores, train_mse, test_mse) 

if __name__ == "__main__":
   
    layer_sizes = [10, 1, 1, 10]
    L2_reg = 0.00000
    param_scale = 1.0
    batch_size = 10 
    num_epochs = 250
    step_size = 0.01
    hyper_iter = 60
    hyper_step_size = 0.05
    hyper_decay = 0.4
    hyper_decay_after = 20 
    hyper_decay_every = 30
    hyper_L2_reg = 0.00001
    rank_loss_scaling_factor = None
    outputFname = '/tmp/results'
    fake_train_images = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    fake_train_labels = np.array([[1.0],[0.0],[0.0],[1.0]])  
    #Validation data (At initialization, it is a copy of training data)
    fake_valid_images = np.copy(fake_train_images)
    fake_valid_labels = np.copy(fake_train_labels)
   
    mse_tolerance = 0.01
    num_petridish_iter = 1 
    num_petridish_points = 20
    num_top_points = 2 #Number of points to be checked in petridish for ground-truth evaluation
    perf_thresh = 0.97 #If performance reaches this mark then stop petridish
    slope_thresh = 0.5 #Do not include points with slope less than this threshold

    #Load the ground truth 
    dir_path = os.path.dirname(os.path.realpath(__file__))+'/'
    ground_truth_fname = dir_path+'mar8_search_slope_full.pkl'
    full_sorted_c_r_list = helper.load_pickle(ground_truth_fname)
    #full_sorted_c_r_list = [(c, r) for c, r in full_sorted_c_r_list if c > 0.35 and c < 0.39]
    #train_sorted_c_r_list = helper.randomly_select_K(full_sorted_c_r_list, perf_thresh, num_petridish_points, slope_thresh)
    train_sorted_c_r_list = [(0.3736363636363636, 0.9676633298397064), (0.3938383838383838, 0.9666700025399526), (0.414040404040404, 0.9661333322525024),  (0.45444444444444443, 0.9645633300145467), (0.4948484848484848, 0.9625533382097881), (0.515050505050505, 0.9617333332697551), (0.5756565656565656, 0.9596799969673157), (0.616060606060606, 0.9568799952665965), (0.6766666666666666, 0.9540533363819123), (0.717070707070707, 0.9526066660881043), (0.818080808080808, 0.9479466597239177), (0.8988888888888888, 0.9435500025749206), (0.9796969696969696, 0.9413000007470449), (1.0605050505050504, 0.9373233377933502), (1.2625252525252524, 0.9289600014686584), (1.3029292929292928, 0.9281666696071624), (1.424141414141414, 0.9239499926567077), (1.525151515151515, 0.921233328183492)]
    #train_sorted_c_r_list = [(0.6362626262626262, 0.9567066649595897), (1.2019191919191918, 0.9321533342202505), (1.2827272727272727, 0.928876664241155), (1.3433333333333333, 0.9264966626962026), (1.3837373737373737, 0.9261333306630453), (1.7271717171717171, 0.9155666649341583)]
    #train_sorted_c_r_list = [(0.45444444444444443, 0.9613), (0.12, 0.957), (1.1211111111111112, 0.9369), (1.5655555555555554, 0.9167), (2.01, 0.9076), (0.01, 0.8999)]#[(0.45444444444444443, 0.9645633300145467), (0.11101010101010099, 0.9554066697756449), (1.121111111111111, 0.9354000012079875), (1.5655555555555554, 0.9201899985472362), (2.01, 0.908623335758845), (0.01, 0.901146666208903)]#, (0.2322, 0.9727)]
    (lines, best_scores, train_mse, test_mse) = main(num_petridish_iter = num_petridish_iter, num_top_points = num_top_points, perf_thresh = perf_thresh, layer_sizes= layer_sizes, L2_reg= L2_reg, param_scale = param_scale, batch_size = batch_size, num_epochs = num_epochs, step_size = step_size, hyper_iter = hyper_iter, hyper_step_size = hyper_step_size, hyper_decay = hyper_decay, hyper_decay_after = hyper_decay_after, hyper_decay_every = hyper_decay_every, hyper_L2_reg = hyper_L2_reg, rank_loss_scaling_factor = rank_loss_scaling_factor, mse_tolerance = mse_tolerance, outputFname = outputFname, train_sorted_c_r_list = train_sorted_c_r_list, full_sorted_c_r_list=full_sorted_c_r_list, fake_train_images = fake_train_images, fake_train_labels = fake_train_labels, fake_valid_images = fake_valid_images, fake_valid_labels = fake_valid_labels)
 
    import datetime
    now = datetime.datetime.now() 
    expt_id = '{:02d}'.format(now.month) + '{:02d}'.format(now.day) + '{:02d}'.format(now.hour) +'{:02d}'.format(now.minute)   
    print ("Results file is results/results_"+expt_id) 
    with open ("results/results_"+expt_id, 'w') as fwr:
        fwr.writelines(lines)
        fwr.writelines(str(best_scores)+"\n")
    #for line in lines:
    #    print (line)

    #print (best_scores)
