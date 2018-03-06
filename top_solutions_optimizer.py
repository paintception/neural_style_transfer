"""
This program computes Gatys et al algorithm for neural style on the same sets of feature maps as the SolutionAnalysis.py one but with the difference that it runs until convergence.
Furthermore it is possible to decide which kind of loss function to use i.e. the one which minimized the style by the total amount of filter or the amount of used feature maps.
The first one will later be used for the simulated annealing optimization algorithm.
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
import glob

from keras import backend as K

from multiprocessing import *

import Learner

import tensorflow as tf
import numpy as np 

SOLUTION_PATH = "/data/msabatelli/style_feature_search/incremental_solutions/random_solutions/"
LOSSES_PATH = "/data/msabatelli/style_feature_search/incremental_solutions/images/munch.jpg/"	# Weird name but normalized losses are here

parser = argparse.ArgumentParser()

parser.add_argument('--content_img', type=str)
parser.add_argument('--style_img', type=str)
parser.add_argument('--normalizer', type=bool)

args = parser.parse_args()

content_img = args.content_img
style_img = args.style_img
normalizer = args.normalizer

if normalizer is True:
    LOSSES_OUTPUT_PATH = "/data/msabatelli/style_feature_search/incremental_solutions/normalized_top_solutions/losses/"
    IMAGES_OUTPUT_PATH = "/data/msabatelli/style_feature_search/incremental_solutions/normalized_top_solutions/images/"

elif normalizer is False:
    LOSSES_OUTPUT_PATH = "/data/msabatelli/style_feature_search/incremental_solutions/classical_top_solutions/losses/"
    IMAGES_OUTPUT_PATH = "/data/msabatelli/style_feature_search/incremental_solutions/classical_top_solutions/images/"

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

def load_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)

    return img

def get_solutions():
    return(tot_solutions) 

def get_losses():
    filelist = glob.glob(os.path.join(LOSSES_PATH, '*.npy'))
    total_losses = list()
    
    for infile in sorted(filelist):
	l = np.load(infile)
	total_losses.append(l)

    return([i for sublist in total_losses for i in sublist])

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

    if normalizer is True:
        N = binary_solution.count(1)
        M = shapes[1]
        
        return K.sum(K.square(S - C)) / (4. * (int(N) ** 2) * (int(M) ** 2))

    else:
        return K.sum(K.square(S - C)) / (4. * (512 ** 2) * (int(M) ** 2))

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

    solution_loss = list()
    
    try:
       os.makedirs(IMAGES_OUTPUT_PATH + "solution_" + str(cnt) + "/") 
    except OSError as e:
       pass
        
    for i in range(50):
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)
       	solution_loss.append(min_val)
        img = deprocess_image(x.copy())
        fname = result_prefix + "_" + '_at_iteration_%d.png' % i
	
	if i % 10 == 0:
            imsave(IMAGES_OUTPUT_PATH+"solution_" + str(cnt)+ "/" + fname, img)
        
    return(solution_loss)

x = load_image(content_img)

res_tracker = list()
cnt = 0 

solutions = get_solutions()
losses = get_losses()

highest_loss = np.max(losses)
valuable_solution_threshold = int((highest_loss*80)/100)

for solution, loss in zip(solutions, losses):
    if loss >= valuable_solution_threshold:

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

        try:
            os.makedirs(LOSSES_OUTPUT_PATH + "solution_" + str(cnt) + "/") 
        except OSError as e:
            pass
        
        np.save(LOSSES_OUTPUT_PATH + "solution_" + str(cnt)  + "/" + "normalized_loss"+".npy", final_loss)
        print("Results are saved!")
        cnt += 1
