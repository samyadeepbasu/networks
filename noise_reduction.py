""" Gene Regulatory Network Reconstruction using Pseudo Time Series Data 
	Dataset : Progression of Mouse Embryonic Fibroblast Cells to Myocytes
	Validation of Network : Extracted Network from http://regulatorynetworks.org                                                                      """

#Libraries
import math
import openpyxl as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
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


#Function to standardize the data by assuming a Gaussian Distribution - Center the Data
def standardize(time_measurements):
	#Average / Mean
	mean = np.mean(np.array(time_measurements))

	#Standard Deviations
	std = np.std(np.array(time_measurements))
    
    #Change the array
	new_time = (time_measurements - mean) / std

	return new_time

	
#Function to order cells by pseudo time
def pseudo_time(data_matrix):
	#Open the file corresponding to Pseudo Time Measurement
	time_file = open('data2/time.txt','r')
	
	#Extraction of Pseudo Time from the List
	ordered_cells = [line.split()[1] for line in time_file.readlines()]

	#Convert into Numpy Array
	ordered_cells = np.array(ordered_cells)

	#Conversion from string to float
	new_ordered_cells = [float(item) for item in ordered_cells]

	new_ordered_cells = standardize(new_ordered_cells)
	
	#Get the Indexes for the sorted order
	indexes = np.argsort(new_ordered_cells)
	#print indexes

	#Order the data_matrix
	new_matrix = data_matrix[:,indexes]

	#Convert every expression level in the matrix into floating point number
	new_matrix = new_matrix.astype(float)	

	
	return new_matrix,new_ordered_cells


#Function to get the ground truth for the dataset
def ground_truth():
	#Open and Initialise the File
	g_file = open('ground_truth/stamlab_for_data2.txt','r')

	#Conversion of the interactions in appropriate format
	interactions = [ (int(line.split()[3]),int(line.split()[2])) for line in g_file.readlines()]
	
	return interactions

#Function to draw the gaussian curve
def gaussian(time_array):
	#Mean
	mu = np.mean(time_array)

	#Sigma
	sigma = np.std(time_array)
    
    #Gaussian Distribution Function
	y = np.exp(-np.power(time_array - mu,2.) / 2 * np.power(sigma,2.))    

	plt.scatter(time_array,y)

	plt.show()


#Function to create a state matrix by clustering cells in nearby time points
def state_matrix(ordered_matrix,ordered_cells,no_states):
	#Sorting the cell ordering 
	time_ordered = sorted(ordered_cells)

	time_ordered = np.array([[time] for time in time_ordered])

	clusterer = KMeans(n_clusters=no_states,init='k-means++')

	cluster_labels = clusterer.fit_predict(time_ordered)

	#print cluster_labels

	main_indexes = []

	temp = []

	#Obtaining indexes after clustering
	for i in range(len(cluster_labels)-1):
		if cluster_labels[i] == cluster_labels[i+1]:
			temp.append(i)
			if i == len(cluster_labels) - 2:
				temp.append(i+1)
				main_indexes.append(temp)

		else:
			temp.append(i)
			main_indexes.append(temp)
			temp = []



	state_matrix = []

	for index in main_indexes:
		state = np.sum(ordered_matrix[:,index],axis=1) / len(index)
		state_matrix.append(state)

  
	return state_matrix

#Function to normalise the state matrix to consolidate all gene expression values from 0 to 1
def normalise(state_matrix):
	#Normalise the matrix 
	matrix = np.array([ (item - np.min(item))/(np.max(item) - np.min(item)) for item in state_matrix])

	return matrix

	

def clustered_state(states):
	#Extraction of list of transcription factors and the Gene Expression Matrix
	transcription_factors, data_matrix = create_data_matrix()

	#Order the Matrix by Pseudo Time -- Sequential Data -- Normalized by Time
	ordered_matrix, ordered_cells = pseudo_time(data_matrix)
    
    #Normalisation of Pseudo Time and Visualisation
	#gaussian(ordered_cells)
	new_state_matrix = state_matrix(ordered_matrix,ordered_cells,states)

	normalized_matrix = normalise(new_state_matrix)

	return normalized_matrix



def matrix_pass():
	#Extraction of list of transcription factors and the Gene Expression Matrix
	transcription_factors, data_matrix = create_data_matrix()

	#Order the Matrix by Pseudo Time -- Sequential Data -- Normalized by Time
	ordered_matrix, ordered_cells = pseudo_time(data_matrix)

	return ordered_matrix, ordered_cells



#final_matrix = clustered_state(50)

#print final_matrix