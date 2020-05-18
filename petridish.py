"""A multi-layer perceptron for classification of MNIST handwritten digits."""
from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad
from autograd.core import getval
from autograd.misc.flatten import flatten
#from autograd.misc.optimizers import sgd
#from autograd.misc.optimizers import adam
from opt import adam
from opt import sgd 

from helper import compute_lr
import math

DEBUG = False 

#np.mean does not work for lists in autograd, 
#so writing a custom function
def _mean(score_list): 
    return np.sum(score_list)/len(score_list)

def compute_mean_std(scores):
    mean = 0.0
    std = 0.0
    mean = np.sum(scores)
    mean = mean/len(scores)
    std = np.sum([abs(i - mean)**2 for i in scores])
    std = std/len(scores)
    std = np.sqrt(std)
    
    return (mean, std)
    

def sigmoid(x):
    return 0.5 * (np.tanh(x / 2.) + 1)

#Randomly pick an index to select from a list of initial weights 
def get_init_idx(combined_init_params, mask_params, random_state):
    if len(combined_init_params) != len(mask_params):
        print ('Error in get_init_idx'); import sys; sys.exit()
    idx_list = range(len(combined_init_params))
    idx_list = random_state.permutation(list(idx_list)) #Shuffle is not supported in python 3.x
    init_idx = idx_list[0]
    return init_idx


def init_random_params(scale, layer_sizes, num_archs=1, rs=npr.RandomState(seed=None)):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
    params_list = []
    for i, (m, n) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        
        #Input layer is shared across architectures in the petridish 
        if i==0: 
            if scale==0.0:
                params_list.append((scale * np.ones((m, n*num_archs)),  # weight matrix
                                    scale * np.ones(n*num_archs)))      # bias vector
            else:  
                #print ('Shoudl NOT GET CALLED'); import sys; sys.exit()
                params_list.append((scale * rs.randn(m, n*num_archs),   # weight matrix
                                    scale * rs.randn(n*num_archs)))     # bias vector
        
        #Hidden layers are independent for each architecture in the petridish 
        else: 
            if scale==0.0:
                params_list.append((scale * np.ones((m*num_archs, n*num_archs)),   # weight matrix
                                    scale * np.ones(n*num_archs)))               # bias vector
            else:
                #print ('Shoudl NOT GET CALLED'); import sys; sys.exit()
                params_list.append((scale * rs.randn(m*num_archs, n*num_archs),   # weight matrix
                                    scale * rs.randn(n*num_archs)))               # bias vector
    return params_list


#Creates a combined network consisting of all the petridish architectures
def create_combined_init_params(init_params_list, layer_sizes):
    
    num_archs = len(init_params_list)

    #Initialize a fully-connected combined network and its mask with zero weights 
    zero_init_params = init_random_params(scale=0.0, layer_sizes=layer_sizes, num_archs=num_archs) 
    mask_init_params = init_random_params(scale=0.0, layer_sizes=layer_sizes, num_archs=num_archs) 
    
    combined_init_params = []
    mask_params = []

    #Layer-by-layer set the weights
    for i, ((W1, b1), (W2, b2), (mask_W1, _)) in enumerate(zip(zero_init_params, init_params_list[0], mask_init_params)):
        
        #Layer 0
        if i==0:
            W1 = np.tile(W2, num_archs) #We can repeat the weights of the first layer because it is fully connected
            b1 = np.tile(b2, num_archs)
            mask_W1 = np.ones(W1.shape)
        
        #Layer 1 to N
        #Selectively set the weights to their initial values (Rest are zeroes)
        #This ensures that all architectures in the petridish are independent of each other
        else:
            for j in range(num_archs):
                for k in range(num_archs):
                    if j == k: 
                        rows, cols = W2.shape
                        W1[j*rows:(j+1)*rows, k*cols:(k+1)*cols] = W2
                        mask_W1[j*rows:(j+1)*rows, k*cols:(k+1)*cols] = np.ones(W2.shape) #Set Mask to 1 for valid weights
                        rows,  = b2.shape
                        b1[k*rows:(k+1)*rows] = b2

        combined_init_params.append((W1, b1))
        mask_params.append(mask_W1)
    return combined_init_params, mask_params

def generate_init_params(sorted_c_r_list, param_scale, layer_sizes, init_params, random_state):
    init_params_list = []
    num_archs = 1
    if init_params is None:
        init_params = init_random_params(param_scale, layer_sizes, num_archs, random_state)
    for idx, (c,r) in enumerate(sorted_c_r_list):
        #Should we have same init_params for each c-value or different??
        #For now, we are having same. Think more
        #Also, should they be generated at each hyper-iteration or re-used
        #For now, re-used for each hyper-iteration
        #init_params = init_random_params(param_scale, layer_sizes)
        init_params_list.append(init_params)
    return init_params_list, init_params

def neural_net_predict(params, inputs, factor=0.23):
    """Implements a deep neural network for classification.
       params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       returns normalized class log-probabilities."""
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        #inputs = np.tanh(outputs) #tanh performs better than sigmoid
        inputs = sigmoid(factor*outputs)
    return outputs - logsumexp(outputs, axis=1, keepdims=True)

def neural_net_predict_binary(params, mask_param, inputs, factor_list):
    """Implements a deep neural network for classification.
       params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       returns normalized class log-probabilities."""
    for i, ((W, b), mask_W) in enumerate(zip(params, mask_param)):
        outputs = np.dot(inputs, np.multiply(W, mask_W)) + b  #(BatchSize, Num Hidden)
        if (outputs.shape[1]%len(factor_list)) != 0:
            print ('Error in neural_net_predict_binary'); import sys; sys.exit() 
        num_hidden = (outputs.shape[1])/len(factor_list) #Should be an integer
        #inputs = np.tanh(outputs) #tanh performs better than sigmoid
        if i < len(params)-1:
            inputs = sigmoid(np.multiply(np.repeat(factor_list, num_hidden), outputs))
        else:
            inputs = sigmoid(outputs)
    return  inputs

def l2_norm(params):
    """Computes l2 norm of params by flattening them into a vector."""
    flattened, _ = flatten(params)
    return np.dot(flattened, flattened)

#Returns a list of losses with element representing the loss for each architecture point in the petridish
def log_posterior_binary_test(params, mask_param, inputs, targets, L2_reg, factor_list):
    num_archs = len(factor_list)
    outputs = neural_net_predict_binary(params, mask_param, inputs, factor_list) #Batch size, num_archs/len(factor_list)
    #outputs = outputs.reshape(targets.shape) #Make it a 2D array instead of 1D array in order to match the shape of the targets 
    outputSize = targets.shape[1]
    negative_binary_cross_entropy_list = [np.mean(targets*np.log(outputs[:, i*outputSize:(i+1)*outputSize].reshape(targets.shape)) + (1-targets)*np.log(1-outputs[:, i*outputSize:(i+1)*outputSize].reshape(targets.shape))) for i in range(num_archs)]
    #if DEBUG: [print(loss) for loss in negative_binary_cross_entropy_list]
    return negative_binary_cross_entropy_list 

def log_posterior_binary(params, mask_param, inputs, targets, L2_reg, factor_list):
    log_prior = -L2_reg * l2_norm(params)
    #if DEBUG: print ('Train BCE loss')
    negative_binary_cross_entropy_list = log_posterior_binary_test(params, mask_param, inputs, targets, L2_reg, factor_list)
    total_loss = np.sum(negative_binary_cross_entropy_list) + log_prior
    return total_loss 


def accuracy(params, inputs, targets, factor):
    #target_class    = np.argmax(targets, axis=1)
    predicted_class = np.around(neural_net_predict_binary(params, inputs, factor), decimals=0)
    #print ('Accuracy, True, Predicted: ', targets, neural_net_predict_binary(params, inputs, factor), predicted_class) 
    return np.mean(predicted_class == targets)

def main(layer_sizes= [2, 1, 1], L2_reg= 0.0001, param_scale = 0.1, batch_size = 20, num_epochs = 10, step_size = 1.0, hyper_iter =450, hyper_step_size = 2.5, hyper_decay = 0.333, hyper_decay_after = 300, hyper_decay_every = 300, hyper_L2_reg = 0.000002, rank_loss_scaling_factor = 20.0, mse_tolerance = 0.01, outputFname = '/tmp/results.txt', sorted_c_r_list = None, init_params_list=[], combined_init_params = None, mask_params = None, random_state = None, hyper_train = True, fake_train_images = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), fake_train_labels = np.array([[1.0],[0.0],[0.0],[1.0]]), fake_valid_images = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), fake_valid_labels = np.array([[1.0],[0.0],[0.0],[1.0]])):
    
    #import pickle
    #with open("mnist19_try.pkl", 'rb') as f_rd: 
    #     sorted_c_r_list = pickle.load(f_rd)
    
    if False: #layer_sizes is None:
        import helper
        args = helper.get_config()

        # Model parameters
        layer_sizes = args.layer_sizes
        L2_reg = args.L2_reg

        # Training parameters
        param_scale = args.param_scale #Very important param. Should be low
        batch_size =  args.batch_size #inner loop batch size. Ignore if number of training examples < batch size 
        num_epochs =  args.num_epochs  #inner loop epochs
        step_size =   args.step_size #Very important param. 
    
        #This is the not hyper_epochs. If hyper_iter > num_batches, then batches repeat. 
        #Ideally, we should have hyper_iter = num_batches * hyper_epochs.
        hyper_iter =      args.hyper_iter
        hyper_step_size = args.hyper_step_size
        hyper_L2_reg =    args.hyper_L2_reg
        rank_loss_scaling_factor = args.rank_loss_scaling_factor

    #if DEBUG: print("Loading training data...")
    #N, actual_train_images, actual_train_labels, actual_test_images,  actual_test_labels = load_mnist()
    
        #Following should be float values. Required by autograd. Cannot take gradient w.r.t integers
        fake_train_images = np.array(args.fake_train_images)
        fake_train_labels = np.array(args.fake_train_labels)
        #fake_test_images = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]) 
        #fake_test_labels = np.array([[1.0],      [0.0],      [0.0],      [1.0]])    #np.zeros((10,1))
        #fake_test_images =  np.array([[1.0, 0.0], [0.0, 1.0]])
        #fake_test_labels =  np.array([[0.0],      [0.0]])    #np.zeros((10,1))
    test_batch_size = batch_size

    # Define training objective
    def objective(train_images,train_labels,params, mask_param, c_list, iter):
        num_batches = int(np.ceil(len(train_images) / batch_size))
        def batch_indices(iter):
            idx = iter % num_batches
            return slice(idx * batch_size, (idx+1) * batch_size)
        idx = batch_indices(iter)
        #if DEBUG: print ('In train objective')
        return -log_posterior_binary(params, mask_param, train_images[idx], train_labels[idx], L2_reg, c_list)

    # Define test objective
    def test_objective(params, mask_param, test_images, test_labels, c_list, iter):
        num_batches = int(np.ceil(len(test_images) / test_batch_size))

        def batch_indices(iter):
            idx = iter % num_batches
            return slice(idx * batch_size, (idx+1) * test_batch_size)
        idx = batch_indices(iter)
        if DEBUG: print ('Test BCE loss')
        return [-x for x in log_posterior_binary_test(params, mask_param, test_images[idx], test_labels[idx], L2_reg, c_list)]
    
    def inner_loop(train_images,train_labels, init_params, mask_param, c_list):
        def objective_wrapper(params,iter): #sgd in line 108 magically supplies the iter parameter
            return objective(train_images,train_labels,params, mask_param, c_list, iter)

        objective_grad = grad(objective_wrapper)
        num_batches = int(np.ceil(len(train_images) / batch_size))

        ##if DEBUG: print("     Epoch     |    Train accuracy  |       Test accuracy  ")
        #def print_perf(params, iter, gradient):
        #    if iter % num_batches == 0:
        #        train_acc = accuracy(params, train_images, train_labels, c_list)
        #        test_acc  = accuracy(params, test_images, test_labels, c_list)
                #if DEBUG: print("{:15}|{:20}|{:20}".format(iter//num_batches, train_acc, test_acc))
        # The optimizers provided can optimize lists, tuples, or dicts of parameters.
        #if DEBUG: print("DOING INNER LOOP SGD")
        optimized_params = adam(objective_grad, init_params, step_size=step_size,
                            num_iters=num_epochs * num_batches,callback=None)
       
        #test_loss = test_objective(optimized_params, train_images, train_labels, c, 0)
        return optimized_params
 
    trained_weights_dict = dict()
    test_loss_dict = dict()

    #sorted_c_r_list = [(0.32, 0.95), (1.2, 0.91), (1.6, 0.87)]
    #sorted_c_r_list = [(0.45444444444444443, 0.9613), (0.12, 0.957), (1.1211111111111112, 0.9369), (1.5655555555555554, 0.9167), (2.01, 0.9076), (0.01, 0.8999)]#, (0.2322, 0.9727)]
    #c_1 = 0.2322
    #c_2 = 0.32
    def check_success(scores):
        sorted_scores = []
        for idx, val in enumerate(scores):
            sorted_scores.append((idx, val))
        sorted_scores.sort(key=lambda x: x[1], reverse=True)
        for idx, val in enumerate(sorted_scores):
            if (idx != val[0]):
                if DEBUG: print ('Failed')
                return False
        if DEBUG: print ('Success in rank')
        return True
        
    #init_params_list = generate_init_params(sorted_c_r_list, param_scale, layer_sizes) 
 
    def test_loss(train_images, init_params, mask_param, c_list, t):
        [train_images, valid_images] = np.split(train_images, 2) #Extracting train and valid images
        trained_weights = inner_loop(train_images,fake_train_labels, init_params, mask_param, c_list)
        return (test_objective(trained_weights, mask_param, valid_images, fake_valid_labels, c_list, t), trained_weights)
    
    def pairwise_rankloss(score_1, score_2, target):
        scaling_factor = rank_loss_scaling_factor 
        if (score_1 > score_2):        
            scaling_factor = 0.0
        diff = (score_1 - score_2)*scaling_factor
        rank_loss = -target*diff + np.log(1 + np.exp(diff))
        #print ('Score1, Score2, Rank Loss', score_1, score_2, rank_loss)
        return rank_loss 

    def rankloss(scores):

        total_rank_loss = 0.0
        rank_loss_count = 0
        for i in range(len(scores)-1):
            for j in range(i, len(scores)):
                if i !=j:
                    score_1 = scores[i]
                    score_2 = scores[j]
                    target = 1.0 #This is assuming scores are already placed in their actual/target index (based on sorted_c_r_list ordering) 
                                 #1.0 means score_1 > score_2 and 0.0 means score_1 < score_2
                    rank_loss = pairwise_rankloss(score_1, score_2, target) 
                    total_rank_loss += rank_loss
                    rank_loss_count += 1 #Total number of rank comparisons
                    if DEBUG: print ('Pairs', i, j, 'Score', score_1, score_2, rank_loss)
        total_rank_loss = total_rank_loss/rank_loss_count
        if DEBUG: print ('Total Rank Loss', total_rank_loss)
        return total_rank_loss
    
    def mse_loss(scores):
        mean, std =  compute_mean_std(scores)
        normalized_pred_scores = [ (x-mean)/std for x in scores] #Whiten the data
        for index, (i, j) in enumerate(zip(scores, normalized_pred_scores)):
            if DEBUG: print (sorted_c_r_list[index][0], 'Predicted scores', i, 'Normalized Pred scores', j)
        
        true_scores = [ r for (c,r) in sorted_c_r_list ] #Convert accuracy to error
        mean, std =  compute_mean_std(true_scores)
        normalized_true_scores = [ (x-mean)/std for x in true_scores] 
        for index, (i, j) in enumerate(zip(true_scores, normalized_true_scores)):
            if DEBUG: print (sorted_c_r_list[index][0], 'True scores', i, 'Normalized True scores', j)
        mse = np.mean(np.array([(i-j)**2 for i, j in zip(normalized_true_scores, normalized_pred_scores)]))
        if DEBUG: print ('Mean Squared Error', mse)
        #print ('True Scores and Normalized scores', true_scores, normalized_true_scores)
        #print ('Pred Scores and Normalized scores', scores, normalized_pred_scores)
        return mse 
        

    def get_scores(train_images, t):
        score_list = [] #Scores will be appended in the same order as the sorted_c_r_list
        trained_weights_list = []
        #for idx, (c, r) in enumerate(sorted_c_r_list):
        init_idx = get_init_idx(combined_init_params, mask_params, random_state)
        c_list = [c for (c, r) in sorted_c_r_list]
        test_loss_list, trained_weights_list = test_loss(train_images, combined_init_params[init_idx], mask_params[init_idx], c_list, t) #Reuse init_params 
        score_list = [-loss for loss in test_loss_list] #Validation loss needs to be inverted to get rank score
        #score_list.append(score)
        #trained_weights_list.append(trained_weights)
        return score_list, trained_weights_list

    #Train the network and save the weights
    def hyper_objective(train_images, t): #t is test iteration
        if DEBUG: print ('Hyper iteration', t)
    
        score_list, trained_weights_list = get_scores(train_images, t)
        success_flag = check_success(score_list)
        if success_flag == True:
            if DEBUG: print ('Successful Learnt Images', train_images) 
            #with open (outputFname, 'a') as fwr:
            #    fwr.writelines("Hyperepoch "+str(t)+" Rank order correct "+str(train_images)+"\n")
        if (t==0 or t==hyper_iter-1 or success_flag == True):
            if DEBUG: print ('Hyper Iteration', t, 'scores')
            if DEBUG: [print (score) for score in score_list]
            
        if (t==hyper_iter-1 or success_flag == True):
            if DEBUG==True:
                for idx, trained_weights in enumerate(trained_weights_list):
                    print ('Final Weights', idx, trained_weights)
            #print ('Final trained_images', train_images)
        #Write to output file from inside the petridish. This is to ensure that we capture the result before we lose in the next iter
        mse = mse_loss(score_list) 
        #if mse < mse_tolerance: 
        #    with open (outputFname, 'a') as fwr:
        #        fwr.writelines("Hyperepoch "+str(t)+" MSE Loss converged "+str(mse)+str(train_images)+"\n")
            
        return mse+l2_norm(train_images)*hyper_L2_reg
    
    def get_results(learnt_images, train, iteration=0):
        score_list, trained_weights_list = get_scores(learnt_images, t=iteration) #t is not useful unless number of images > batch_size 
        success_flag = check_success(score_list)
        return success_flag, score_list, trained_weights_list


    if hyper_train == True: 
        hyper_objective_grad = grad(hyper_objective)
        h_i = 0 #Initializing hyper-iteration
        optimized_params = np.concatenate((fake_train_images, fake_valid_images), axis=0) #Valid images and labels have the same dimension as the train images and labels
 
        #Following while loop runs Hyper-training with hyper step size decay
        while h_i < hyper_iter:
            hyper_lr, num_iters = compute_lr(init_lr=hyper_step_size, current_iter=h_i, decay_after=hyper_decay_after, decay_factor=hyper_decay, decay_every=hyper_decay_every) 
            print ('Hyper Learning Rate', hyper_lr, num_iters)
            optimized_params = adam(hyper_objective_grad, optimized_params, step_size=hyper_lr,num_iters=num_iters)
            h_i = h_i + num_iters
        #optimized_params = adam(hyper_objective_grad, optimized_params, step_size=hyper_step_size/3,num_iters=300)
        
        if DEBUG: print ('Learned training images', optimized_params)
        
        success_flag, score_list, trained_weights_list   = get_results(optimized_params, train=hyper_train, iteration=hyper_iter)
        
        mse = mse_loss(score_list) 
        
        #Since the train_images and train_labels have the same dimensionality, 
        #we can split the concatenated optimized_params into two equal parts 
        #Added validation images and labels that have the same dimensionality as the training images and labels
        [optimized_train_images, optimized_valid_images] = np.split(optimized_params, 2)

        return mse, success_flag, score_list, optimized_train_images, fake_train_labels, optimized_valid_images, fake_valid_labels 
    
    #Test. Only return scores
    else:
       
        optimized_params = np.concatenate((fake_train_images, fake_valid_images), axis=0) #Valid images and labels have the same dimension as the train images and labels
       
        success_flag, score_list, trained_weights_list = get_results(optimized_params, train=hyper_train, iteration=0)
        
        mse = mse_loss(score_list) 
        
        #Since the train_images and train_labels have the same dimensionality, 
        #we can split the concatenated optimized_params into two equal parts 
        #Added validation images and labels that have the same dimensionality as the training images and labels
        [optimized_train_images, optimized_valid_images] = np.split(optimized_params, 2)

        return mse, success_flag, score_list, optimized_train_images, fake_train_labels, optimized_valid_images, fake_valid_labels 


if __name__ == '__main__':
    
    main()


