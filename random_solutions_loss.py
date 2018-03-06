"""
Visualization of the results obtained by SolutionAnalysis.py
Per different Images the behaviour of the normalized loss on the different random binary solutions is visualized. 
'"""

import numpy as np 

from matplotlib import pyplot as plt 
from matplotlib import cm

from itertools import groupby
from scipy import mean

import seaborn as sns
import os 
import random
import glob
import time

LOSSES_PATH = "../images/"
IMAGES = ["Image 1", "Image 2", "Image 3", "Image 4"]
COLORS = ["magenta","blue","green","orange"]

plt.style.use('ggplot')

def get_losses():

	total_losses = list()
	
	for root, directories, filenames in os.walk(LOSSES_PATH):
		for directory in directories:
			d = os.path.join(LOSSES_PATH, directory) 
			for root, dirs, files in os.walk(d):  
				artist_losses = list()
				files.sort()
				for filename in files:
					tmp_loss = np.load(str(root) + "/" + filename)
					artist_losses.append(tmp_loss)
					
				flattened_losses = [item for sublist in artist_losses for item in sublist]
				total_losses.append(flattened_losses)
				
	return total_losses
				
def custom_plot(x, y, c, **kwargs):
	ax = kwargs.pop('ax', plt.gca())
	base_line, = ax.plot(x, y, **kwargs)
	ax.fill_between(x, 0.92*y, 1.10*y, facecolor=c,alpha=200)

def plot_results(average_losses):
	
	abs_max = max(map(lambda x: x[3], average_losses))

	normalized_outputs = list()

	for loss in average_losses:
		normalized = [float(i)/max(loss) for i in loss]
		normalized_outputs.append(normalized)

	colors = cm.rainbow(np.linspace(0, 1, len(normalized_outputs)))

	for loss, img, c in zip(normalized_outputs, IMAGES, COLORS):
		
		y = range(len(loss))
		custom_plot(y, np.asarray(loss), c, lw=1)

		plt.xlabel('Amount of Feature Maps')
		plt.ylabel("Normalized "+r"$\mathcal{L}(\vec p, \vec a, \vec x)$")
		plt.plot(loss, label=img, color=c)
		plt.legend(loc="upper left")
	
	plt.show()
	
def main():
	total_losses = get_losses()
	plot_results(total_losses)

if __name__ == '__main__':
	main()