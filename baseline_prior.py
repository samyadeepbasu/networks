""" Baseline Algorithm to Classify Regulatory Edges as being true or false  
	 Considerations : Steady State and Pseudo Time Series States   
	 Task : Sequential Classification        
	 Models Implemented : 1. KNN with Dynamic Time Warping
																	  """

############################################################################################
############################################################################################


#Libraries
import math
import numpy as np
import random
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics.pairwise import rbf_kernel,polynomial_kernel,sigmoid_kernel,laplacian_kernel,chi2_kernel
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor, GradientBoostingRegressor
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import networkx as nx
from numpy import inf
from scipy.stats import pearsonr, spearmanr
from scipy import spatial
from sklearn.metrics import mutual_info_score
from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from random import shuffle


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

	#Conversion of the interactions in appropriate format  -- Target,Regulator
	interactions = [ (int(line.split()[2]),int(line.split()[3])) for line in g_file.readlines()]

	for item in interactions: #Remove Self-Loops
		if item[0] == item[1]:
			interactions.remove(item)
	
	return interactions


#Function to get a list of times
def time():
	#Time Measurements
	time_series = open('data2/time.txt','r')

	times = [line.split()[1] for line in time_series.readlines()]

	#times = order_time()  #Using Imputation Algorithm and TSCAN

	sorted_time = np.sort(np.array(times).astype(float))

	#Normalise
	#normalised_time = (sorted_time - np.min(sorted_time)) / (np.max(sorted_time) - np.min(sorted_time))
	normalised_time = sorted_time / max(sorted_time)
	#print normalised_time


	return normalised_time

#Function to order cells by pseudo time
def pseudo_time(data_matrix):
	#Open the file corresponding to Pseudo Time Measurement
	time_file = open('data2/time.txt','r')
	
	#Extraction of Pseudo Time from the List
	ordered_cells = [line.split()[1] for line in time_file.readlines()]

	#ordered_cells = order_time() #Using Imputation Algorithm and TSCAN

	#Convert into Numpy Array
	ordered_cells = np.array(ordered_cells)

	#Conversion from string to float
	new_ordered_cells = [float(item) for item in ordered_cells]
	
	#Get the Indexes for the sorted order
	indexes = np.argsort(new_ordered_cells)

	#Order the data_matrix
	new_matrix = data_matrix[:,indexes]

	#Convert every expression level in the matrix into floating point number
	new_matrix = new_matrix.astype(float)	
	
	return new_matrix


#Function to construct feature sequences for each gene interaction
def construct_features(interactions, ordered_matrix):
	#Number of States 
	no_states = len(ordered_matrix[0])

	#Feature Vector
	feature_vector = []

	for edge in interactions:
		#Target Gene
		target = ordered_matrix[edge[0]]
		#Regulator Gene
		regulator = ordered_matrix[edge[1]]
		
		temp = []

		for i in range(0,no_states-2):
			#Check for 5 conditions
			if regulator[i+1] > regulator[i] and target[i+2] > target[i+1]:
				temp.append(0)

			elif regulator[i+1] > regulator[i] and target[i+2] < target[i+1]:
				temp.append(1)

			elif regulator[i+1] < regulator[i] and target[i+2] > target[i+1]:
				temp.append(2)

			elif regulator[i+1] < regulator[i] and target[i+2] < target[i+1]:
				temp.append(3)

			else:
				temp.append(4)

		feature_vector.append(temp)
		

	#Convert into numpy array
	feature_vector = np.array(feature_vector)

	return feature_vector	


#Function to create feature vectors for false connections
def false_features(interactions,ordered_matrix):
	temp = []
	for edge in interactions:
		temp.append(edge[0])
		temp.append(edge[1])
	
	#Unique Transcription Factors
	unique_nodes = list(set(temp))

	false_vectors = []

	for i in unique_nodes:
		for j in unique_nodes:
			if i != j:
				if (i,j) not in interactions:
					false_vectors.append((i,j))

	vectors = construct_features(false_vectors,ordered_matrix)	

	return vectors

#Function to get the minimum of three numbers
def find_minimum(a,b,c):
	minimum_list = []
	minimum_list.append(a)
	minimum_list.append(b)
	minimum_list.append(c)

	min_list = sorted(minimum_list)

	return min_list[0]


#Function to Compute the Distance Matrix
def distance_matrix (total_set):
	X = np.array([vector[0] for vector in total_set])

	#Initialise Matrix
	dist_matrix = np.zeros((len(X),len(X)))

	for i in range(0,len(X)):
		for j in range(i+1,len(X)):
			#dist_matrix[i][j] = DTW_distance(X[i],X[j])
			print i
			dist_matrix[i][j], path = fastdtw(X[i].reshape(1,-1),X[j].reshape(1,-1), dist=euclidean)


	return dist_matrix


#Function to put the different states into bins
def binning(state_matrix, times, k): #Time Passed is sorted
	#Cluster the time points
	cluster = AgglomerativeClustering(n_clusters=k)
	
	#Convert into matrix for K-means
	time_matrix = np.array([[time] for time in times])
	
	#Labels for the clusters
	labels = cluster.fit_predict(time_matrix)
	
	#Unique Labels
	ranges = []
	for i in range(0,len(labels)-1):
		if labels[i+1] != labels[i]:
			ranges.append(i)

	start = 0
	end = len(labels)

	#Create Bins for Clustering the states
	bins = []

	for i in range(len(ranges)):
		bins.append((start,ranges[i]+1))
		start = ranges[i]+1
	
	#Append the last one
	bins.append((start,end))

	""" Averaging out the states """
	
	new_state_matrix = []
	total = 0
	for bin in bins:
		temp_matrix = state_matrix[bin[0]:bin[1]]
		#Average out the expression levels for each gene
		temp = []

		
		for i in range(0,len(temp_matrix[0])):
			temp.append(np.mean(temp_matrix[:,i]))


		new_state_matrix.append(temp)


	
	new_state_matrix = np.array(new_state_matrix)

	""" Create a new time matrix """

	new_time_matrix = []

	for bin in bins:
		new_time_matrix.append(times[bin[0]])


	return new_state_matrix, new_time_matrix



#Function to train the model using KNN + DTW
def train_model(training, testing):
	""" Algorithm : For each edge in the testing set, create a distance matrix & extract the data point to which it has the Smallest Distance """
	
	predictions = []
	
	i = 0

	for edge in testing:
		signal_vector = edge[0]

		distances = []

		for vec in training:
			training_signal_vector = vec[0]
			dist ,path = fastdtw(signal_vector.reshape(1,-1),training_signal_vector.reshape(1,-1), dist=euclidean)

			distances.append(dist)


		distances = np.array(distances)

		sorted_indexes = np.argsort(distances)

		predictions.append(training[sorted_indexes[0]][1])

		print i 
		i += 1

	#print "#"
	predictions = np.array(predictions)
	true_labels = np.array([edge[1] for edge in testing])

	score = roc_auc_score(true_labels,predictions)

	return score

 
def main():
	tf, data_matrix = create_data_matrix()

	data_matrix = data_matrix.astype(float)

	data_matrix = pseudo_time(data_matrix)

	time_series = time()

	main_matrix = data_matrix.copy()

	AUC_total = []

	for k in range(15,60):
		data_matrix, time_ordered = binning(main_matrix.transpose(),time_series,25)

		data_matrix = data_matrix.transpose()	

		interactions = ground_truth()

		feature_vectors_true = construct_features(interactions, data_matrix)

		y1 = np.repeat(1,len(feature_vectors_true))

		feature_vectors_false = false_features(interactions,data_matrix)

		y2 = np.repeat(0,len(feature_vectors_false))

		Y = np.concatenate((y1,y2))

		X = np.concatenate((feature_vectors_true,feature_vectors_false),axis=0)	

		#Generate Random Numbers
		total_set = [(X[i],Y[i]) for i in range(0,len(X))]

		#Shuffle the list
		shuffle(total_set)

		#DTW_matrix = distance_matrix(total_set)

		training_set = total_set[0: int(len(total_set)*0.8)]

		testing_set = total_set[int(len(total_set)*0.8):]

		AUC_score = train_model(training_set,testing_set)

		AUC_total.append(AUC_score)

	

	print AUC_total
	print max(AUC_total)
	X = range(0,len(AUC_total))
	plt.plot(X,AUC_total)
	plt.show()
	
	return




main()


