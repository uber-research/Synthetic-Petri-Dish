import argparse
import numpy as np

#Returns the learning rate and the number of iterations to run ADAM/SGD using that learning rate 
def compute_lr(init_lr, current_iter, decay_after, decay_factor, decay_every):
    lr = init_lr * (decay_factor ** max(0, np.ceil((current_iter-(decay_after-1.0))/(decay_every*1.0))))
    if current_iter == 0:
        num_iters = decay_after
    elif current_iter >= decay_after:
        num_iters = decay_every
    else:
        print ("Error in Compute LR, Exiting")
        import sys; sys.exit()
    return lr, num_iters

def plot_lr():
    from matplotlib import pyplot as plt
    import config
    from random import random
    max_iter = 600
    plot_count = 0
    for h_s_s in config.hyper_step_size:
     for h_d in config.hyper_decay:
      for h_d_a in config.hyper_decay_after:
       for h_d_e in config.hyper_decay_every:
        
        if (h_s_s <= 0.1 and h_d < 0.9) or (h_s_s >= 2.0 and h_d > 0.5):
            print ('Ignoring config', h_s_s, h_d)
            continue
        plot_count += 1
        plot_label = str(h_s_s) + '_' + str(h_d) + '_' + str(h_d_a) + '_' + str(h_d_e)
        lr_list = []
        i = 0
        while i < max_iter:
            lr, num_iters = compute_lr(init_lr = h_s_s, current_iter=i, decay_after=h_d_a, decay_factor=h_d, decay_every=h_d_e)
            i = i + num_iters
            lr_list.extend([lr]*num_iters)
        plt.plot (range(max_iter), lr_list, color=np.random.rand(3,), label=plot_label)
    print ('Total Plots', plot_count)
    plt.legend()
    plt.show()
   
def load_pickle(fname):
    import pickle
    import sys
    with open (fname, 'rb') as f_rd:
        if (sys.version_info[:3] > (3,0)):
                data = pickle.load(f_rd, encoding='latin1')
        else:
                data = pickle.load(f_rd)
    return data

#Randomly initialize init parameters for each hyper-iteration
def get_init_param_list(total_iter, c_r_list, param_scale, layer_sizes, init_params, random_state):
    import petridish
    combined_init_params = []
    mask_params = []
    for _ in range (total_iter):
        init_params_list, init_params = petridish.generate_init_params(c_r_list, param_scale, layer_sizes, init_params, random_state) #Note: for now, all petridish networks have the same weight init at a given hyper-iter
        temp_combined_init_params, temp_mask_params = petridish.create_combined_init_params(init_params_list, layer_sizes)
        combined_init_params.append(temp_combined_init_params)
        mask_params.append(temp_mask_params)

    return combined_init_params, mask_params, init_params

def save_pickle(fname, data):
    import pickle
    with open (fname, 'w') as f_wr:
        pickle.dump(data, f_wr)
    return

#Randomly sample K points from the list of tuples. 
def randomly_select_K(sorted_c_r_list, perf_thresh, K, slope_thresh=0.0):
    import random
    while True:
        idx_list = random.sample(range(len(sorted_c_r_list)), K)
        sorted_idx_list = sorted(idx_list)
        selected_c_r_list = []
        for idx in sorted_idx_list:
            if (sorted_c_r_list[idx][1] >= perf_thresh) or (sorted_c_r_list[idx][0] < slope_thresh): #If we selected one of the best points, then ignore and resample
                    break
            else:
                selected_c_r_list.append(sorted_c_r_list[idx])
        if len(selected_c_r_list) == K:
            return selected_c_r_list


def extract_top_arch(score_list, sorted_c_r_list, K=2):
    
    sorted_score_c_r_list = [x for _,x in sorted(zip(score_list,sorted_c_r_list), reverse=True)]
    
    #Extract top K from full_sorted_c_r_list based on their respective scores in full_score_list
    return sorted_score_c_r_list[0:K] 

def extract_worst_arch(score_list, sorted_c_r_list, K=1):
    
    sorted_score_c_r_list = [x for _,x in sorted(zip(score_list,sorted_c_r_list), reverse=True)]
    
    #Extract worst K from full_sorted_c_r_list based on their respective scores in full_score_list
    return sorted_score_c_r_list[-K:] 

def add_new_arch_ground_truth(sorted_c_r_list, new_c_r_list):
    import bisect
    for (new_c, new_r) in new_c_r_list:
 
        for i, (c, r) in enumerate(sorted_c_r_list):
            index = i
            if (new_r > r):
                break
            else:           #Case where we did not find any place to insert. Append at the end 
                index = i+1
        sorted_c_r_list.insert(index, (new_c, new_r))
    
    return sorted_c_r_list

#When petridish converges (even before hyper_iter are done), the results
#are dumped in a text file. We read the text file to extract the results
#The first instance of convergence is returned. If no convergence, return None
def read_images_from_file(fname, num_images, checkStr):
    with open (fname, 'r') as f_rd:
        data = f_rd.readlines()
    image_matrix = []
    found = False
    for line in data:
        row = []
        if checkStr in line or found == True:
            found = True
            line = line.split('[')[-1]
            line.replace('[', '')
            line = line.strip(']\n')
            words = line.strip(' ').split(' ')
            for w in words:
                if w is not '':
                    row.append(float(w))
            image_matrix.append(row) 
        #Return if we completed reading the first match
        if len(image_matrix) == num_images:  
            return np.array(image_matrix)
    return None
            

if __name__=="__main__":
    plot_lr()

def get_config():
    
    parser = argparse.ArgumentParser(description='Starting Evolve LSTM experiment')
    
    # Model parameters
    parser.add_argument('-layer_sizes', '--layer_sizes',
                        dest='layer_sizes',
                        help='Size of the petri-dish network',
                        nargs="*",
                        default=[2, 1, 1],
                        type=int,
                        required=False)
    parser.add_argument('-L2_reg', '--L2_reg',
                        dest='L2_reg',
                        help='L2 weight penalty in inner-loop',
                        default='0.0001',
                        type=float,
                        required=False)
    
    #Inner-loop training parameter
    parser.add_argument('-param_scale', '--param_scale', #Very important param. Should be low
                        dest='param_scale',
                        help='Scale of initial weights for the petri-dish network',
                        default='0.1',
                        type=float,
                        required=False)
    parser.add_argument('-batch_size', '--batch_size', #Ignore of training examples < batch size??
                        dest='batch_size',
                        help='Batch size',
                        default='20', #Set a large number so that batch size is the number of training examples
                        type=int,
                        required=False)
    parser.add_argument('-num_epochs', '--num_epochs', 
                        dest='num_epochs',
                        help='Num epochs',
                        default='10', 
                        type=int,
                        required=False)
    parser.add_argument('-step_size', '--step_size',  
                        dest='step_size',
                        help='Learning Rate',
                        default='1.0',               #Important parameter
                        type=float,
                        required=False)

    #Outer-loop/Hyper-loop parameters 
    parser.add_argument('-hyper_iter', '--hyper_iter', #This is the not hyper_epochs. If hyper_iter > num_batches, then batches repeat. 
                        dest='hyper_iter',             #Ideally, we should have hyper_iter = num_batches * hyper_epochs. 
                        help='Hyper iteration',
                        default='450', 
                        type=int,
                        required=False)
    parser.add_argument('-hyper_step_size', '--hyper_step_size',  
                        dest='hyper_step_size',
                        help='Hyper Learning Rate',
                        default='2.5',               #Important parameter
                        type=float,
                        required=False)
    parser.add_argument('-hyper_L2_reg', '--hyper_L2_reg',
                        dest='hyper_L2_reg',
                        help='Hyper L2 weight penalty in inner-loop',
                        default='0.000002',
                        type=float,
                        required=False)
    parser.add_argument('-rank_loss_scaling_factor', '--rank_loss_scaling_factor',
                        dest='rank_loss_scaling_factor',
                        help='Scaling factor for rank loss',
                        default='20.0',
                        type=float,
                        required=False)

    parser.add_argument('-fake_train_images', '--fake_train_images',
                        dest='fake_train_images',
                        help='Fake Train images',
                        nargs="*",
                        default=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                        type=float,
                        required=False)
    parser.add_argument('-fake_train_labels', '--fake_train_labels',
                        dest='fake_train_labels',
                        help='Fake Train Labels',
                        nargs="*",
                        default=[[1.0],      [0.0],      [0.0],      [1.0]],
                        type=float,
                        required=False)
    
    args = parser.parse_args()
    return args 

