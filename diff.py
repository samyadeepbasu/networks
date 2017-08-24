""" Modelling Expression using Differential Equation """


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
from noise_reduction import clustered_state
from sklearn.linear_model import Lars


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
	
	#Get the Indexes for the sorted order
	indexes = np.argsort(new_ordered_cells)

	#Order the data_matrix
	new_matrix = data_matrix[:,indexes]

	#Convert every expression level in the matrix into floating point number
	new_matrix = new_matrix.astype(float)	
	
	return new_matrix

#Extract the Top Regulators for Each Gene
def create_model(state_matrix,transcription_factors):
	regulators = {}

	for i in range(0,len(transcription_factors)):
		#Create the training set
		X = []
		y = []
		for j in range(1,len(state_matrix)):
			#Append the expression level of the previous step
			X.append(state_matrix[j-1].tolist())

			#The output value is the difference / rate of change of expression
			y.append(state_matrix[j][i] - state_matrix[j-1][i])

        #Copy the list of Transcription Factors
		tf_copy = list(transcription_factors)

		#Remove the current transcription factor
		tf_copy.remove(tf_copy[i])

		#Remove the corresponding column from the training set
		[expression.remove(expression[i]) for expression in X]

		""" Feature Selection using Random Forests """

		#Initialise the model using Random Forests and Extract the Top Regulators for each Gene
		forest_regressor = RandomForestRegressor(n_estimators = 2,criterion = 'mse')

		#Fit the training data into the Model
		forest_regressor.fit(X,y)

		#Extract the important features corresponding to a particular gene
		important_features = forest_regressor.feature_importances_

		#Regulators for the Network
		regulators[transcription_factors[i]] = important_features	
		

	
	return regulators


def main():
	#Transcription Factors and Data Matrix
	transcription_factors, data_matrix = create_data_matrix()

	#Order the cells using Pseudo Time
	ordered_matrix = pseudo_time(data_matrix)

	#Transpose the matrix 
	state_matrix = ordered_matrix.transpose()
    
    #Churn out the top regulators from each gene
	regulators = create_model(state_matrix,transcription_factors)




	



main()