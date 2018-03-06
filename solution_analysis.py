"""
The main idea here is to get insights about which feature maps in the different layers of the Inception VGG19 neural network are representative for a particular artistic style.
The algorithm consists in an adaptation of Gatys et al. neural style algorithm with appropriately modified loss functions.

The experiment consists in randomly sampling a set of feature maps from the last layer of the neural network and run the neural style algorithm. The sampling occurs from a set of random solutions that consist
in binary solutions of length 512 (which corresponds to the maximum amount of possible feature maps). Every solution makes use of one more randomly selected feature map when compared to the previous one, we start from a 
solution which makes use of only a single feature map until one which uses them all.

In order to have insights about the impact of the feature map the style loss function needs to be changed. We replace the amount of filters that are used by the amount of feature maps. In such a way we are able to get a normalized
style loss function and measure the impact of different sets of feature maps. This estimation is given by 2 backpropagations of the algorithm, the mean is then stored as a reliable estimate for the impact of a particular set of feature maps.   

The arguments of the program consist in a content image, a style image and a batch interval which samples the different random solutions from the total set of them. This makes it computationally more efficient to compute the gradients
for each sampled binary solution.
"""

from __future__ import print_function
from __main__ import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from keras.applications import vgg19
from keras.preprocessing.image import load_img, img_to_array

from scipy.misc import imsave

from collections import OrderedDict
from operator import itemgetter

from scipy.optimize import fmin_l_bfgs_b

import time
import argparse
import random
import copy
import math
import gc

from keras import backend as K

from multiprocessing import *

import Learner

import tensorflow as tf
import numpy as np 

SOLUTION_PATH = "/data/msabatelli/style_feature_search/incremental_solutions/random_solutions/"
IMAGE_OUTPUT_PATH = "/data/msabatelli/style_feature_search/incremental_solutions/images/"
LOSSES_OUTPUT_PATH = "/data/msabatelli/style_feature_search/incremental_solutions/losses/"

parser = argparse.ArgumentParser()
parser.add_argument('--start_batch', type=int)
parser.add_argument('--end_batch', type=int)
parser.add_argument('--content_img', type=str)
parser.add_argument('--style_img', type=str)

args = parser.parse_args()

start = args.start_batch 
end = args.end_batch
content_img = args.content_img
style_img = args.style_img

width, height = load_img(content_img).size
img_nrows = 200
img_ncols = int(width * img_nrows / height)
content_weight = 0.025
style_weight = 1.0
total_variation_weight = 1.0
result_prefix = "output"

tot_solutions = np.load(SOLUTION_PATH + "incremental_solutions.npy")

class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        
        return grad_values

def get_solution(total_style_reference_features, total_style_combination_features, binary_solution):

    single_feat_solution = list()
    single_ref_solution = list()

    for idx, bit in enumerate(binary_solution):
        if bit == 1:
            feat_bit = total_style_reference_features[:,:, idx]
            ref_bit = total_style_combination_features[:,:, idx]

            single_feat_solution.append(feat_bit)
            single_ref_solution.append(ref_bit)

    new_features_solutions = (tf.stack(single_feat_solution, axis=2))
    new_reference_solutions = (tf.stack(single_ref_solution, axis=2))

    return(new_features_solutions, new_reference_solutions, binary_solution)

def estimate_probability(pr):
    return(random.random() < pr)

def load_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)

    return img

def initialize_loss():
    return K.variable(0.)

def get_random_content_features(layer_features):
    
    total_content_reference_features = layer_features[0, :, :, :]
    total_combination_content_features = layer_features[2, :, :, :]

    return(total_content_reference_features, total_combination_content_features)

def gram_matrix(x):
    assert K.ndim(x) == 3
    
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    
    return gram

def prepare_style_loss(style, combination, binary_solution):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3

    S = gram_matrix(style)
    C = gram_matrix(combination)

    sl = original_style_loss(S, C, style, binary_solution)

    return sl

def original_style_loss(S, C, style, binary_solution):

    shapes = style.get_shape()

    N = binary_solution.count(1)
    M = shapes[1]
    
    return K.sum(K.square(S - C)) / (4. * (int(N) ** 2) * (int(M) ** 2))

def original_content_loss(base, combination):
    return K.sum(K.square(combination - base))

def total_variation_loss(x):
    assert K.ndim(x) == 4
 
    a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
    b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
 
    return K.sum(K.pow(a + b, 1.25))

def eval_loss_and_grads(x):
    x = x.reshape((1, img_nrows, img_ncols, 3))
    outs = f_outputs([x])
    loss_value = outs[0]

    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    
    return loss_value, grad_values

def deprocess_image(x):
    x = x.reshape((img_nrows, img_ncols, 3))
    
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')

    return x

def run_experiment(x, cnt):

    batch_loss = list()

    for i in range(2):
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)
       	batch_loss.append(min_val)
        #print('Current loss value:', min_val)
        img = deprocess_image(x.copy())
        fname = result_prefix + "_" + str(cnt) + '_at_iteration_%d.png' % i
	
    	#print("Storing Image: ", fname)

        imsave(IMAGE_OUTPUT_PATH+fname, img)
        #print("Image is Saved!")

    return(sum(batch_loss)/len(batch_loss))

x = load_image(content_img)

solutions = tot_solutions[start:end]

print("Processing Interval:", start, end)

res_tracker = list()
cnt = 0

for solution in solutions:

    loss = initialize_loss()
    content_layer_features = Learner.outputs_dict['block5_conv2']   #Keep it the same!
    
    random_content_features = get_random_content_features(content_layer_features)

    sampled_base_image_features = random_content_features[0]
    sampled_combination_image_features =random_content_features[1]
    
    loss += content_weight * original_content_loss(sampled_base_image_features, sampled_combination_image_features)   
    
    for layer_name in Learner.feature_layers:
	
	style_layer_features = Learner.outputs_dict[layer_name]

        total_style_reference_features = style_layer_features[1, :, :, :]     # Set corresponding to the total pool of features
                                                                         # our aim is to find the optimal subset in here
        total_style_combination_features = style_layer_features[2, :, :, :]
       
        starting_set = get_solution(total_style_reference_features, total_style_combination_features, solution)

        starting_feature_maps = starting_set[0]
        starting_reference_maps = starting_set[1]

        binary_solution = starting_set[2]

        sl = prepare_style_loss(starting_feature_maps, starting_reference_maps, list(binary_solution))
        loss += (style_weight / len(Learner.feature_layers)) * sl                    

    loss += total_variation_weight * total_variation_loss(Learner.combination_image)

    grads = K.gradients(loss, Learner.combination_image)

    outputs = [loss]

    if isinstance(grads, (list, tuple)):
	outputs += grads
    else:
        outputs.append(grads)

    f_outputs = K.function([Learner.combination_image], outputs)

    evaluator = Evaluator()
    final_loss = run_experiment(x, cnt)        

    print("===========")
    print(final_loss)
    print("===========")    

    res_tracker.append(final_loss)
 	
    cnt = cnt + 1

try:
    os.makedirs(LOSSES_OUTPUT_PATH + str(style_img) + "/") 
except OSError as e:
    pass
	    
np.save(LOSSES_OUTPUT_PATH + str(style_img) + "/" + "_loss_batch_"+str(end)+".npy", res_tracker)
print("Results are saved!")