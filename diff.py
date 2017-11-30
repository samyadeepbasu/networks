""" Baseline Learning Model for reconstruction of regulatory networks using Gradient Boosting  """


#Libraries
import math
import openpyxl as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.decomposition import FactorAnalysis
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor, GradientBoostingRegressor
from sklearn import metrics
import networkx as nx
from noise_reduction import clustered_state
from sklearn.linear_model import Lars
from sklearn.metrics import mutual_info_score
from numpy import inf
from order import order_time
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

#Function to Clean the data and create a training set
def create_data_matrix():
	#Open the file for transcription factors
	tf_file = open("data3/tf.txt","r")

	#Transcription Factors List
	tf_list = [factor[:len(factor)-1] for factor in tf_file.readlines()]

	#Gene Expression Matrix creation
	exp_file = open("data3/data.txt","r")
	
	#Split the lines into list from the file and storage in list
	data_matrix = [row[:len(row)-1].split('\t') for row in exp_file.readlines()]	
	
	#Conversion into numpy array
	data_matrix_array = np.array(data_matrix)

	return tf_list, data_matrix_array


#Function to order cells by pseudo time
def pseudo_time(data_matrix):
	#Open the file corresponding to Pseudo Time Measurement
	time_file = open('data3/time.txt','r')
	
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


#Function to get a list of times
def time():
	#Time Measurements
	time_series = open('data3/time.txt','r')

	times = [line.split()[1] for line in time_series.readlines()]

	#times = order_time()  #Using Imputation Algorithm and TSCAN

	sorted_time = np.sort(np.array(times).astype(float))

	#Normalise
	#normalised_time = (sorted_time - np.min(sorted_time)) / (np.max(sorted_time) - np.min(sorted_time))
	normalised_time = sorted_time / max(sorted_time)
	print normalised_time


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


		#Compute Mutual information

		""" Feature Selection using Random Forests / Extra Trees / Gradient Boosting Regressor """

	
		#Initialise the model using Random Forests and Extract the Top Regulators for each Gene
		#forest_regressor = DecisionTreeRegressor(random_state = 0)
		forest_regressor = RandomForestRegressor(n_estimators = 1,criterion = 'mse')
		#forest_regressor = ExtraTreesRegressor(n_estimators = 700 ,criterion = 'mse')   #Extra Trees - Randomized Splits

		#forest_regressor = GradientBoostingRegressor(loss='ls',learning_rate=0.09,n_estimators=1200,random_state=42) #Gradient Boosting with least square


		#Fit the training data into the Model
		forest_regressor.fit(X,y)

		#Extract the important features corresponding to a particular gene
		important_features = forest_regressor.feature_importances_  

		#Regulators for the Network
		regulators[transcription_factors[i]] = important_features	
		

	
	return regulators


def mutual_information_model(state_matrix,transcription_factors):

	regulators = {}
	#For each gene compute a vector with the mutual information values
	for i in range(0,len(transcription_factors)):

		positions = range(0,len(transcription_factors))

		positions.remove(i)

		feature_importances = []

		for element in positions:
			#Compute Mutual information
			feature_importances.append(mutual_info_score(state_matrix[i],state_matrix[element]))


		feature_importances = np.array(feature_importances)

		regulators[transcription_factors[i]] = feature_importances




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
	g_file = open('ground_truth/stamlab_for_data3.txt','r')

	#Conversion of the interactions in appropriate format  -- Target,Regulator
	interactions = [ (int(line.split()[2]),int(line.split()[3])) for line in g_file.readlines()]

	for item in interactions:
		if item[0] == item[1]:
			interactions.remove(item)
	
	return interactions


#Function to generate the AUPR Graph and Score / AUC Graph and Score
def get_graph(edges,ground_truth,actual_tf):

	temp = []
	#print actual_tf
	#print len(temp)
	for i in range(0,len(edges)):
		if edges[i][0] in actual_tf and edges[i][1] in actual_tf:
			temp.append(edges[i])
	
	common = 0
	
	#Replace edges with temp
	for e in ground_truth:
		for edge in temp:
			if e[0] == edge[0] and e[1]==edge[1]:
				common += 1

	
	if len(temp) == 0 :
		return 0,0,0
	
	precision = float(common) / len(temp)
	recall = float(common) / len(ground_truth)
	fpr = (float(len(temp)) - float(common)) / float(len(actual_tf)*(len(actual_tf)-1) - len(ground_truth))

	if fpr>1:
		print len(temp)
		print common
		print "-#-"

	return precision,recall,fpr
	


#Function to normalise the state matrix to consolidate all gene expression values from 0 to 1
def normalise(state_matrix):
	#Normalise the matrix 
	#matrix = np.array([ (item - np.min(item))/(np.max(item) - np.min(item)) for item in state_matrix])

	#Normalise along the columns
	temp_matrix = state_matrix.transpose()

	#matrix = np.array([ (item - np.min(item))/(np.max(item) - np.min(item)) for item in temp_matrix])

	for i in range(len(temp_matrix)):
		temp_matrix[i] = temp_matrix[i] / max(temp_matrix[i])

	matrix = temp_matrix.transpose()


	return matrix

#Normalise the gene expression to normal distribution
def normal(data_matrix):
	temp_matrix = data_matrix.transpose()
	#print len(temp_matrix)
	for i in range(0,len(temp_matrix)):
		#Mean
		mean = np.mean(temp_matrix[i])
		std = np.std(temp_matrix[i])
		temp_matrix[i] = (temp_matrix[i] - mean) / std

	temp_matrix = temp_matrix.transpose()
	
	return temp_matrix


#Area under Curve
def area(fpr,recall):

	return metrics.auc(fpr,recall)


""" Binning for creating clusters """

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


def find_tf(tfs):
	real_interactions = ground_truth()
	
	temp = []
	for interaction in real_interactions:
		temp.append(interaction[0])
		temp.append(interaction[1])
		
		
	
	#Unique Transcription Factors
	temp = list(set(temp))
   
	return temp


def main():
	#Transcription Factors and Data Matrix
	transcription_factors, data_matrix = create_data_matrix()

	#Transcription Factors - Unique
	tf_no_edges = find_tf(transcription_factors)	

	
	#Normalise the matrix
	normalised_state_matrix =  data_matrix.astype(float)  #normal(state_matrix)#normalise(state_matrix)		
	
	#Churn out the top regulators from each gene
	#regulators = create_model(normalised_state_matrix,transcription_factors,time_series)
	#regulators = create_model_LARS(normalised_state_matrix,transcription_factors)
	regulators = mutual_information_model(normalised_state_matrix,transcription_factors)
	
	start = 1

	precisions = []
	recalls = []
	fprs = []

	i = start

	while i <=9890: #9900
		threshold = get_threshold(regulators,i)

		edges = top_regulators(regulators,threshold,transcription_factors)

		real_interactions = ground_truth()

		precision,recall,fpr = get_graph(edges,real_interactions,tf_no_edges)

		if precision == recall == fpr == 0 :
			break

		precisions.append(precision)
		recalls.append(recall)
		fprs.append(fpr)
		#print i 
		i += 8



	a = area(np.array(fprs),np.array(recalls))

	b = area(np.array(recalls),np.array(precisions))
	
	print a 
	print b

	#AUROC Curve
	#plt.figure(1)
	#plt.plot(np.array(fprs),np.array(recalls))
	#plt.show()

	fig, ax = plt.subplots()
	ax.plot(np.array(fprs),np.array(recalls), c='black')
	line = mlines.Line2D([0, 1], [0, 1], color='red')
	transform = ax.transAxes
	line.set_transform(transform)
	ax.add_line(line)
	plt.show()

	
	return a,b 

	
	
def get_total():
	AUC = []
	AUPR = []
	for i in range(2):
		auc, aupr = main()

		AUC.append(auc)
		AUPR.append(aupr)


	print AUC
	print AUPR
	


#get_total()
main()