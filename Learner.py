import os

from keras.applications import vgg19
from keras.preprocessing.image import load_img, img_to_array

from keras import backend as K

import numpy as np 

import random

def load_image(image_path, img_nrows, img_ncols):
	img = load_img(image_path, target_size=(img_nrows, img_ncols))
	img = img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = vgg19.preprocess_input(img)

	return img

def get_base_image(img_nrows, img_ncols):
	return K.variable(load_image("../images/mat.jpg", img_nrows, img_ncols))

def get_style_image(img_nrows, img_ncols):
	return K.variable(load_image("../images/munch.jpg", img_nrows, img_ncols))

def combine_into_tensor(base_image, style_reference_image, combination_image):
	return K.concatenate([base_image, style_reference_image, combination_image], axis=0)

def load_model(input_tensor):
	return vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)

def get_all_feature_layers(model):
	return dict([(layer.name, layer.output) for layer in model.layers])

def main():

	global model
	global outputs_dict
	global feature_layers
	global base_image
	global style_reference_image
	global combination_image
	global input_tensor
	global outputs_dict
	
	feature_layers = ['block4_conv1',
                  'block5_conv1']	# Keep it the same!

	width, height = load_img("../images/mat.jpg").size	# Size of the content image
	img_nrows = 200

	img_ncols = int(width * img_nrows / height) 

	base_image = get_base_image(img_nrows, img_ncols)
	style_reference_image = get_style_image(img_nrows, img_ncols)

	combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

	input_tensor = combine_into_tensor(base_image, style_reference_image, combination_image) 

	model = load_model(input_tensor)
	
	outputs_dict = get_all_feature_layers(model)

	for layer_name in feature_layers:
    	    layer_features = outputs_dict[layer_name]
	    style_reference_features = layer_features[1, :, :, :]


main()
