""" Baseline Algorithm to Classify Regulatory Edges as being true or false  
	 Considerations : Steady State and Pseudo Time Series States   
	 Task : Sequential Classification        
	 Models Implemented : 1. KNN with Dynamic Time Warping
	                      2. SVM
	                      3. LSTM
	                      4. CNN                                                """

############################################################################################
############################################################################################


import math 
import numpy as np 
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

#Function to calculate the DTW distance matrix and churn out the similarity score -- Optimal Match between two sequences
def DTW_matrix(seq_A, seq_B):
	#Length of Sequence A
	len1 = len(seq_A)

	#Length of Sequence B
	len2 = len(seq_B)
    
    #Initialise DTW Matrix
	DTW = np.zeros((len1+1,len2+1))

	for i in range(1,len2+1):
		DTW[0][i] = inf

	for j in range(1,len1+1):
		DTW[j][0] = inf


	#Fill up the DTW Matrix
	for i in range(1,len1+1):
		for j in range(1,len2+1):
			# Current Distance + Minimum Cost from earlier computations
			DTW[i][j] = math.pow((seq_A[i-1] - seq_B[j-1]),2) + find_minimum(DTW[i-1][j-1],DTW[i-1][j],DTW[i][j-1])	

	print DTW
	return DTW[len1][len2]


#Function to create a model and train it -> SVM, LSTM, CNN
def train_model(X,Y):
	a = DTW_matrix(np.array([1,0,1,1]),np.array([0,0,1,1]))


	return

def main():
	tf, data_matrix = create_data_matrix()

	ordered_matrix = pseudo_time(data_matrix)

	interactions = ground_truth()

	feature_vectors_true = construct_features(interactions, ordered_matrix)

	y1 = np.repeat(1,len(feature_vectors_true))

	feature_vectors_false = false_features(interactions,ordered_matrix)

	y2 = np.repeat(0,len(feature_vectors_false))

	Y = np.concatenate((y1,y2))

	X = np.concatenate((feature_vectors_true,feature_vectors_false),axis=0)	

	train_model(X,Y)









	
	return



main()


