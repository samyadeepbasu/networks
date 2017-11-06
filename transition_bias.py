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
	g_file = open('ground_truth/stamlab_for_data2.txt','r')

	#Conversion of the interactions in appropriate format  -- (Regulator --->  Target)
	interactions = [ (int(line.split()[3]),int(line.split()[2])) for line in g_file.readlines()]

	for item in interactions: #Remove Self-Loops
		if item[0] == item[1]:
			interactions.remove(item)
	
	return interactions


#Function to find the unique transcriptional factors 
def find_unique_tfs(interactions):
	temp = []

	for interaction in interactions:
		temp.append(interaction[0])
		temp.append(interaction[1])

	temp = list(set(temp))

	return temp


#Function to generate the negative samples
def get_negative_interactions(positive_interactions):
	#Find the unique transcription factors
	unique_tfs = find_unique_tfs(positive_interactions)

	total_sample = []

	for regulator in unique_tfs:
		for target in unique_tfs:
			if target != regulator:
				total_sample.append((regulator,target))


	negative_samples = list(set(total_sample) - set(positive_interactions))

	return total_sample, negative_samples

#Function to split into training and testing sample
def split(total_samples):
	
	number_edges = len(total_samples)
	
	number_testing = int(0.2*number_edges)

	#Do a random shuffle
	random.shuffle(total_samples)

	chunks = []
	
	for i in range(0,len(total_samples),number_testing):
		chunks.append(total_samples[i:i+number_testing])


	return chunks	

#Function to train the model using Stochastic Gradient Descent and Biasing via Transition Probability
def train_model(W,targets,edge,lam):
	#Get the sum of the predictions of all edges 
	p = 0

	p_positive = 0
	p_negative = 0

	for e in targets:
		prediction = np.matmul(W,e[0].reshape(1,-1).transpose())[0][0]
		p += prediction

		if e[1]==0:
			p_negative += prediction

		elif e[1] == 1:
			p_positive += prediction


	if p_positive > p_negative:
		#No regularization required
		return 2*W

	else: #This is the condition where regularization is required
		positive_derivative = 0
		negative_derivative = 0

		X_positive = 0
		X_negative = 0
		X_total = 0

		for e in targets:
			X_total += e[0]
			if e[1] == 1:
				X_positive += e[0]


			if e[1] == 0:
				X_negative += e[0]


		positive_derivative = (p*X_positive - p_positive*X_total) / (p*p)
		negative_derivative = (p*X_negative - p_negative*X_total) / (p*p)

		#Regularization term
		regularization_term = positive_derivative - negative_derivative

		gradient = 2*W[0] +  lam*regularization_term

		return gradient.reshape(1,-1)



	return 

#Function to create the classification model 
def create_classification_model(training_set,testing_set,expression_matrix):
	#True classification labels
	true_labels = [edge[2] for edge in testing_set]

	edge_scores = []

	truth = []

	regulators = list(set([edge[0] for edge in training_set]))

	""" Go to each local region of the graph and train the model """
	for regulator in regulators:
		targets = [(expression_matrix[edge[1]],edge[2]) for edge in training_set if edge[0]==regulator]

		X = [target[0] for target in targets]
		Y = [target[1] for target in targets]

		#Initialise the weight matrix
		W = np.random.rand(1,len(X[0]))

		epochs = 800

		alpha = 0.001

		#print np.matmul(W,X[0].reshape(1,-1).transpose())

		for i in range(0,epochs):
			#Do a SGD after each edge
			for edge in targets:
				#Update the weight
				gradient = train_model(W,targets,edge,0.1)

				W = W - alpha * gradient

				

			loss = 0
			#Compute Loss
			for edge in targets:
				loss += math.pow((np.matmul(W,edge[0].reshape(1,-1).transpose())[0][0] - edge[1]),2)


		

		#Go to the local testing edges in the local region of the graph and predict scores
		for e in testing_set:
			if e[0] == regulator:
				truth.append(e[2])
				prediction = 

			







			


		
		

		break




	




	return 


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

	positive_samples = [link + (1,) for link in positive_interactions]
	negative_samples = [link + (0,) for link in negative_interactions]

	total_samples = positive_samples + negative_samples

	#Randomly Shuffle the samples
	random.shuffle(total_samples)

	#Gene Expression Matrix
	main_matrix = data_matrix.copy()

	#Split into Training and Testing Data
	splitted_sample = split(total_samples)
	splitted_sample = splitted_sample[:5]

	for i in range(0,len(splitted_sample)):
		testing_set = splitted_sample[i]
		training_set_index = range(0,len(splitted_sample))
		training_set_index.remove(i)

		training_set = []

		for index in training_set_index:
			training_set += splitted_sample[index]


		#auc, aupr,const_aupr = create_model(training_set,testing_set,main_matrix,10)
		auc = create_classification_model(training_set,testing_set,main_matrix)

		break





	return




main()