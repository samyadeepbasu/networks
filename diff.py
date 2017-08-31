""" Baseline Learning Model for reconstruction of regulatory networks using Gradient Boosting  """


#Libraries
import math
import openpyxl as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.decomposition import FactorAnalysis
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor, GradientBoostingRegressor
from sklearn import metrics
import networkx as nx
from noise_reduction import clustered_state
from sklearn.linear_model import Lars
from numpy import inf


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


#Function to get a list of times
def time():
	#Time Measurements
	time_series = open('data2/time.txt','r')

	times = [line.split()[1] for line in time_series.readlines()]

	sorted_time = np.sort(np.array(times).astype(float))

	#Normalise
	normalised_time = (sorted_time - np.min(sorted_time)) / (np.max(sorted_time) - np.min(sorted_time))

	return normalised_time


#Extract the Top Regulators for Each Gene
def create_model(state_matrix,transcription_factors,time_series):
	regulators = {}

	for i in range(0,len(transcription_factors)):
		#Create the training set
		X = []
		y = []
		for j in range(1,len(state_matrix)):
			#Append the expression level of the previous step
			#X.append(state_matrix[j-1].tolist())
			X.append((state_matrix[j-1]*(time_series[j] - time_series[j-1])).tolist())  #Taking dt into account

			#The output value is the difference / rate of change of expression
			y.append(state_matrix[j][i] - state_matrix[j-1][i])

		#Copy the list of Transcription Factors
		tf_copy = list(transcription_factors)

		#Remove the current transcription factor
		tf_copy.remove(tf_copy[i])

		#Remove the corresponding column from the training set
		[expression.remove(expression[i]) for expression in X]

		""" Feature Selection using Random Forests / Extra Trees """

		#Initialise the model using Random Forests and Extract the Top Regulators for each Gene
		#forest_regressor = RandomForestRegressor(n_estimators = 700,criterion = 'mse')
		#forest_regressor = ExtraTreesRegressor(n_estimators = 1000 ,criterion = 'mse')   #Extra Trees - Randomized Splits

		forest_regressor = GradientBoostingRegressor(loss='ls',learning_rate=0.08,n_estimators=1300) #Gradient Boosting with Friedman MSE

		#Fit the training data into the Model
		forest_regressor.fit(X,y)

		#Extract the important features corresponding to a particular gene
		important_features = forest_regressor.feature_importances_

		#Regulators for the Network
		regulators[transcription_factors[i]] = important_features	
		

	
	return regulators


#Function to implement the Least Angle Regressional Model : Assumption=> Genes are regulated by a linear combination
#Extract the Top Regulators for Each Gene
def create_model_LARS(state_matrix,transcription_factors):
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

		""" Feature Selection using Least Angle Regression """

		#Initialise the model using Least Angle Regression
		lars = Lars()

		#Fit the training data into the Model
		lars.fit(X,y)

		#Extract the important features corresponding to a particular gene
		coefficients = lars.coef_

		#Regulators for the Network
		regulators[transcription_factors[i]] = coefficients	
		

	
	return regulators



#Function to extract the threshold
def get_threshold(regulators,t):
	threshold = []

	for key,value in regulators.items():
		threshold += value.tolist()


	sorted_threshold = sorted(threshold)

	return sorted_threshold[t]


def top_regulators(regulators,threshold,transcription_factors):

	edges = []

	
	for key,value in regulators.items():
		#Regulation Values for the current gene
		reg = value

		#Transcription Factors
		tf_temp = list(transcription_factors)
		tf_temp.remove(key)

		temp = [tf_temp[j] for j in range(0,len(reg)) if reg[j]>threshold]

		indexes = [(transcription_factors.index(key),transcription_factors.index(tf)) for tf in temp]

		edges += indexes		
		#print len(reg)



	return edges

#Function to get the ground truth for the dataset
def ground_truth():
	#Open and Initialise the File
	g_file = open('ground_truth/stamlab_for_data2.txt','r')

	#Conversion of the interactions in appropriate format  -- Target,Regulator
	interactions = [ (int(line.split()[2]),int(line.split()[3])) for line in g_file.readlines()]
	
	return interactions


#Function to generate the AUPR Graph and Score / AUC Graph and Score
def get_graph(edges,ground_truth):
	
	common = 0

	for e in ground_truth:
		for edge in edges:
			if e[0] == edge[0] and e[1]==edge[1]:
				common += 1

	#
	
	precision = float(common) / len(edges)
	recall = float(common) / len(ground_truth)
	fpr = (float(len(edges)) - float(common)) / float(9900 - len(ground_truth))

	return precision,recall,fpr
	


#Function to normalise the state matrix to consolidate all gene expression values from 0 to 1
def normalise(state_matrix):
	#Normalise the matrix 
	#matrix = np.array([ (item - np.min(item))/(np.max(item) - np.min(item)) for item in state_matrix])

	#Normalise along the columns
	temp_matrix = state_matrix.transpose()

	matrix = np.array([ (item - np.min(item))/(np.max(item) - np.min(item)) for item in temp_matrix])

	matrix = matrix.transpose()

	return matrix

#Area under Curve
def area(fpr,recall):

	return metrics.auc(fpr,recall)


#Input Dropouts with average expression in the cell
def impute(matrix):	
	mean = []
	count = 0
	for i in range(0,len(matrix)):
		#Get the average of the expressions
		mean_expression = np.mean(matrix[i])

		#Replace zeros with average expression values
		matrix[i][matrix[i] == 0] = mean_expression

		count += np.count_nonzero(matrix[i] == 0)


	return matrix


""" Binning for creating clusters """

def binning(state_matrix, times, k): #Time Passed is sorted
	#Cluster the time points
	cluster = KMeans(n_clusters=k,init='k-means++')
	
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




def main():
	#Transcription Factors and Data Matrix
	transcription_factors, data_matrix = create_data_matrix()

	#Order the cells using Pseudo Time
	ordered_matrix = pseudo_time(data_matrix)

	#Transpose the matrix 
	state_matrix = ordered_matrix.transpose()

	time_series = time()	

	state_matrix, time_series = binning(state_matrix,time_series,270)

	#Impute in the matrix
	#state_matrix = impute(state_matrix)
	
	#Conversion into log
	state_matrix = np.log(state_matrix)

	#Replace -infinity values with zero    
	state_matrix[state_matrix == -inf] = 0

	#print state_matrix
	
	#Normalise the matrix
	normalised_state_matrix = normalise(state_matrix)	

	
	
	#Churn out the top regulators from each gene
	regulators = create_model(normalised_state_matrix,transcription_factors,time_series)

	
	start = 1

	precisions = []
	recalls = []
	fprs = []

	i = start

	while i <=9890:
		threshold = get_threshold(regulators,i)

		edges = top_regulators(regulators,threshold,transcription_factors)

		real_interactions = ground_truth()

		precision,recall,fpr = get_graph(edges,real_interactions)

		precisions.append(precision)
		recalls.append(recall)
		fprs.append(fpr)
		print i 
		i += 8

	print area(np.array(fprs),np.array(recalls))

	print area(np.array(recalls),np.array(precisions))
	
	#AUROC Curve
	#plt.figure(1)
	plt.plot(np.array(fprs),np.array(recalls))
	plt.show()

	#AUPR Curve
	plt.plot(np.array(recalls),np.array(precisions))
	plt.show()



	


	return



	



main()