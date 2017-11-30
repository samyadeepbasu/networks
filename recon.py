""" Gene Regulatory Network Reconstruction using Pseudo Time Series Data and an Ensemble of Random Forests, Lasso Regression & Stability Selection 
	Baseline Method 
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


#Function to create the equations for the model : Parameter k => Number of cells in one states
def create_states(ordered_matrix,transcription_factors,k):
	#New State Matrix
	state_matrix = []
	temp = 0
	i = 0
	
	#Generation of 
	while i < len(ordered_matrix[0]):
		temp = []
		
		#Handling the limit for Edge Cases
		if i+k < len(ordered_matrix[0]):
			limit = i+k
		else:
			limit = len(ordered_matrix[0])

		for j in range(i,limit):
			temp.append(j)

		#Extraction of the relevant columns
		current_state = np.sum(ordered_matrix[:,temp],axis=1)
		
		#State Matrix
		state_matrix.append(current_state.tolist())
			

		i += k


	return np.array(state_matrix)




#Function to create the Baseline Model for extracting top regulators for each gene from Time Series Ordered Measurements
def create_model(state_matrix,transcription_factors):
	#For each transcription factors extract the top regulators with respect to the previous state
	regulators = {}

	for i in range(len(transcription_factors)):
		#Declaration for training set for the Target Gene
		X = []
		y = []

		for j in range(1,len(state_matrix)):
			X.append(state_matrix[j-1].tolist())
			y.append(state_matrix[j][i] - state_matrix[j-1][i])


		#Initialise the model using Random Forests and Extract the Top Regulators for each Gene
		forest_regressor = RandomForestRegressor(n_estimators = 300,criterion = 'mse')

		#Fit the training data into the Model
		forest_regressor.fit(X,y)

		#Extract the important features corresponding to a particular gene
		important_features = forest_regressor.feature_importances_
		
		#Add to the dictionary
		regulators[transcription_factors[i]] = important_features



	return regulators


#Function to create a Baseline Model using Least Angle Regression
def create_model_LARS(state_matrix,transcription_factors):
	regulators = {}

	for i in range(len(transcription_factors)):
		#Declaration for training set for the Target Gene
		X = []
		y = []

		for j in range(1,len(state_matrix)):
			X.append(state_matrix[j-1].tolist())
			y.append(state_matrix[j][i] - state_matrix[j-1][i])


		#Initialise the LARS Model
		lars = Lars()

		#Fit the training data into the Model
		lars.fit(X,y)

		#Extract the important features corresponding to a particular gene
		coefficients = lars.coef_
		
		#Add to the dictionary
		regulators[transcription_factors[i]] = coefficients


	return regulators



#Function to get the ground truth for the dataset
def ground_truth():
	#Open and Initialise the File
	g_file = open('ground_truth/stamlab_for_data3.txt','r')

	#Conversion of the interactions in appropriate format  -- (Regulator,Target)
	interactions = [ (int(line.split()[3]),int(line.split()[2])) for line in g_file.readlines()]
	
	return interactions



#Function to obtain the area under curve
def calculate_area(precision,recall):
	#area = np.trapz(recall,precision)
	area = metrics.auc(recall,precision)
  
	return area


#Function to generate the PR scores
def generate_PR_score(regulators,transcription_factors,real_interactions):
	#Top Regulators for Each Gene
	top_regulators = []

	thresholds = []

	precision = []
	recall = []
	fpr = []

	#Start threshold 
	threshold = 0.00005


	for i in range(0,500):
		top_regulators = []
		
		for gene,value in regulators.items(): 

			score_array = regulators[gene]

			numbers = map((lambda x: np.where(score_array == x) if x>threshold else None ),score_array)

			#Remove None from the list
			regulator_index = [x[0][0] for x in numbers if x is not None]
			
			topmost_regulators = [(i,transcription_factors.index(gene)) for i in regulator_index]

			top_regulators += topmost_regulators



		common = 0

		for element in top_regulators:
			for ele in real_interactions:
				if element[0] == ele[0] and element[1] == ele[1]:
					common += 1

		precision.append(float(common) / float(len(top_regulators)))
		recall.append(float(common)/float(len(real_interactions)))
		fpr.append(float(len(top_regulators) - common) / float(10000 - len(real_interactions)))

		
		print common


		threshold += 0.00025



	#Conversion into Numpy Array
	precision = np.array(precision)

	recall = np.array(recall)

	fpr = np.array(fpr)

	#area = calculate_area(precision,recall)
	area = calculate_area(recall,fpr)

	print area

	plt.plot(recall,fpr)
	#plt.plot(precision,recall)
	plt.show()   



#Normalise the data wrt the cells
def normalize(ordered_matrix):
	#Normalisation by restricting value between 0 and 1 
	new_matrix = ordered_matrix / ordered_matrix.max(axis=0)

	return new_matrix


#Function to get the minimum coefficient value
def get_min_threshold(regulators):
	#Minimum Threshold Value
	minimum_list = []

	#Maximum Threshold Value
	maximum_list = []

	for key, value in regulators.items():
		minimum_list.append(min(value))
		maximum_list.append(max(value))


	minimum = min(minimum_list)
	maximum = min(maximum_list)

	return minimum,maximum



def main():
	#Extraction of list of transcription factors and the Gene Expression Matrix
	transcription_factors, data_matrix = create_data_matrix()


	""" Commented : States without Clustering cells at the same time points  """

	#Order the Matrix by Pseudo Time
	ordered_matrix = pseudo_time(data_matrix)
	
    #State Matrix Formed by clustering the cells at same time points
	state_matrix = clustered_state(370)

	#Fit the model using Random Forests and Lasso Regression -- Returns a dictionary containing feature score for each gene
	regulators = create_model(state_matrix,transcription_factors)
	

	""" Lars """

    #Ground Truth for comparison
	real_interactions = ground_truth()
	
	#Get the Precision Recall scores for different threshold values and parameters
	scores = generate_PR_score(regulators,transcription_factors,real_interactions)	
	



#Main Function Call
main()



