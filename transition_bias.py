""" Algorithm to reconstruct links in a graph by biasing the transition probabilities 
    Basic Idea : Bias the initial transitions in a random walk so that the jumps are into the positive sections of the local regions of the graph  """


#Libraries
import math
import numpy as np
import random
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.metrics.pairwise import rbf_kernel,polynomial_kernel,sigmoid_kernel,laplacian_kernel,chi2_kernel
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor, GradientBoostingRegressor
from sklearn import metrics
import networkx as nx
from numpy import inf
from scipy.stats import pearsonr, spearmanr
from scipy import spatial
from sklearn.metrics import mutual_info_score
from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from sklearn import svm
import tensorflow as tf 
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge,ElasticNet
import networkx as nx



#Function to Clean the data and create a training set
def create_data_matrix():
	#Open the file for transcription factors
	tf_file = open("data2/tf.txt","r")

	#Transcription Factors List
	tf_list = [factor[:len(factor)-1] for factor in tf_file.readlines()]

	#Gene Expression Matrix creation
	exp_file = open("data2/data.txt","r")
	
	#Split the lines into list from the file and storage in list
	data_matrix = [row[:len(row)-1].split('\t') for row in exp_file.readlines()]	
	
	#Conversion into numpy array
	data_matrix_array = np.array(data_matrix)

	return tf_list, data_matrix_array




#Function to get the ground truth for the dataset
def ground_truth():
	#Open and Initialise the File
	g_file = open('ground_truth/stamlab_for_data3.txt','r')

	#Conversion of the interactions in appropriate format  -- (Regulator --->  Target)
	interactions = [ (int(line.split()[3]),int(line.split()[2])) for line in g_file.readlines()]

	for item in interactions: #Remove Self-Loops
		if item[0] == item[1]:
			interactions.remove(item)
	
	return interactions


#Function 
def main():
	#Transcriptional Factors along with Data Matrix
	tf, data_matrix = create_data_matrix()

	#Data Matrix
	data_matrix = data_matrix.astype(float)

	#Positive Samples
	positive_interactions = ground_truth()

	#Ground Truth : Negative Interactions : (Regulator, Target)
	total_samples, negative_interactions = get_negative_interactions(positive_interactions)	
	
	#Randomly shuffle the lists before splitting into training and testing set
	random.shuffle(total_samples)
	random.shuffle(positive_interactions)
	random.shuffle(negative_interactions)

	

	return




main()