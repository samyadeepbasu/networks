""" Link Prediction as a Matrix Factorization Problem for Gene Regulatory Network """


import numpy as np
import random
from sklearn.metrics import mutual_info_score


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


#Function to fill up the matrix before factorization
def distance_information(a,b):
	return mutual_info_score(a,b)


#Function to find the unique transcriptional factors 
def find_unique_tfs(interactions):
	temp = []

	for interaction in interactions:
		temp.append(interaction[0])
		temp.append(interaction[1])

	temp = list(set(temp))

	return temp


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

#Function to fill the matrix
def fill_matrix(main_matrix,data_matrix,unique_tfs):
	for i in range(0,len(unique_tfs)):
		for j in range(0,len(unique_tfs)):
			if i!=j:
				main_matrix[i][j] = distance_information(data_matrix[unique_tfs[i]],data_matrix[unique_tfs[j]])


	return 

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


#Function to manipulate the matrix and empty out the position of the testing set
def manipulate_matrix(filled_matrix,unique_tfs,testing_set):
	for edge in testing_set:
		filled_matrix[unique_tfs.index(edge[0])][unique_tfs.index(edge[1])] = 0


	return filled_matrix


#Function to create a factorization model
def create_factorization_model(training_matrix,k,N):
	""" Split the matrix into two parts and train using GD to optimize the loss function """

	# K : Parameter for Matrix Factorization

	P = np.random.rand(N,k)
	Q = np.random.rand(N,k)

	#Number of steps for Gradient Descent -- In each turn the whole matrix will be updated
	epoch = 5000

	#Learning Rate
	learning_rate = 0.0002

	for i in range(0,len(training_matrix)):
		for j in range(0,len(training_matrix[0])):
			prediction = 0

			#Predict only for those positions in the training data
			if training_matrix[i][j] > 0:
				for K in range(0,k):
					prediction += P[i][K]*Q[K][j]


				#Compute the error
				error = training_matrix[i][j] - prediction

				#Update for the Gradient Descent Step
				








	


	return

#Function 
def main():
	tf, data_matrix = create_data_matrix()

	data_matrix = data_matrix.astype(float)

	#Ground Truth : Positive Interactions : (Regulator, Target)
	positive_interactions = ground_truth()
	
	#Ground Truth : Negative Interactions : (Regulator, Target)
	total_samples, negative_interactions = get_negative_interactions(positive_interactions)	
	
	#Randomly shuffle the lists before splitting into training and testing set
	random.shuffle(total_samples)
	random.shuffle(positive_interactions)
	random.shuffle(negative_interactions)

	#Unique Transcriptional Factors
	unique_tfs = find_unique_tfs(positive_interactions)

	#Compute the matrix
	main_matrix = np.zeros((len(unique_tfs),len(unique_tfs)))

	#Ground Truth Matrix populated with initial scores
	fill_matrix(main_matrix,data_matrix,unique_tfs)

	#Remove information pertaining to test set

	#Split into Training and Testing Data
	splitted_sample = split(total_samples)
	splitted_sample = splitted_sample[:5]
	
	AUC = []
	AUPR = []
	const = []

	#for k in range(2,12):
	for i in range(0,len(splitted_sample)):
		testing_set = splitted_sample[i]
		training_set_index = range(0,len(splitted_sample))
		training_set_index.remove(i)

		training_set = []

		for index in training_set_index:
			training_set += splitted_sample[index]


		training_matrix = manipulate_matrix(main_matrix.copy(),unique_tfs,testing_set)

		completed_matrix = create_factorization_model(training_matrix,10,len(unique_tfs))

		break

























	return




main()


