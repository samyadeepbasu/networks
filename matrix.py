""" Link Prediction as a Matrix Factorization Problem by integrating expression data for Gene Regulatory Network """


import numpy as np
import random
from sklearn.metrics import mutual_info_score
from sklearn import metrics
from sklearn.metrics.pairwise import rbf_kernel,polynomial_kernel,sigmoid_kernel,laplacian_kernel,chi2_kernel
from scipy.stats import pearsonr, spearmanr
from scipy import spatial
import math
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt 

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
	#return spearmanr(a,b)[0]
	#return pearsonr(a,b)[0]
	#return rbf_kernel(a.reshape(1,-1),b.reshape(1,-1))


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
def fill_matrix(main_matrix,total_samples,unique_tfs):
	"""
	for i in range(0,len(unique_tfs)):
		for j in range(0,len(unique_tfs)):
			if i!=j:
				main_matrix[i][j] = distance_information(data_matrix[unique_tfs[i]],data_matrix[unique_tfs[j]])
	"""

	for edge in total_samples:
		main_matrix[unique_tfs.index(edge[0])][unique_tfs.index(edge[1])] = edge[2]


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
def create_factorization_model(training_matrix,k,N,data_matrix,unique_tfs):
	""" Split the matrix into two parts and train using Stochastic Gradient Descent to optimize the loss function """

	# K : Parameter for Matrix Factorization

	P = np.random.rand(N,k)
	#Q = np.random.rand(N,k)
	Q = P.transpose()

	#Number of steps for Gradient Descent -- In each turn the whole matrix will be updated
	epochs = 20

	#Learning Rate
	learning_rate = 0.001

	beta = 0.05

	for steps in range(0,epochs):
		for i in range(0,len(training_matrix)):
			for j in range(0,len(training_matrix[0])):
				prediction = 0

				#Predict only for those positions in the training data
				if training_matrix[i][j] != 0:
					for K in range(0,k):
						prediction += P[i][K]*Q[K][j]


					#Add Mutual Information (Fm(xi,xj)) ---> Node Attribute Information
					sim_score = distance_information(data_matrix[unique_tfs[i]],data_matrix[unique_tfs[j]])
					#prediction *= sim_score
					prediction += distance_information(data_matrix[unique_tfs[i]],data_matrix[unique_tfs[j]])

					#Compute the error --> Non-linear transformation of prediction
					error = training_matrix[i][j] - math.tanh(prediction) #math.tanh(prediction)

					#Update for the Gradient Descent Step
					for K in range(0,k):
						P[i][K] = P[i][K] + learning_rate * (sim_score*error*Q[K][j] - beta*P[i][K])
						#Q[i][K] = P[i][K] + learning_rate * (error * Q[K][j] - beta*P[i][K])
						#P[K][j] = Q[K][j] + learning_rate * (error * P[i][K] - beta*Q[K][j])
						Q[K][j] = Q[K][j] + learning_rate * (sim_score*error* P[i][K] - beta*Q[K][j])



		#Compute the loss here
		updated_matrix = np.matmul(P,Q)

		loss = 0

		for i in range(0,len(training_matrix)):
			for j in range(0,len(training_matrix[0])):
				#If the position is in the training set
				if training_matrix[i][j] != 0:
					#Compute the loss after update
					positional_value = 0

					regularization = 0

					for K in range(0,k):
						positional_value += P[i][K]*Q[K][j]
						regularization += pow(P[i][K],2) + pow(Q[K][j],2)

					loss += 0.5 * pow((training_matrix[i][j] - positional_value),2) + (beta/2) * (regularization)

		print loss


	return P,Q


""" Evaluation Scheme for the model """
#Function to generate AUC and AUPR curves
def generate_graph(edge_scores,testing):
	#Minimum score 
	min_score = min(edge_scores)

	#Maximum Score
	max_score = max(edge_scores)
	
	current_network = [(edge[0],edge[1]) for edge in testing]

	positive_edges = [(edge[0],edge[1]) for edge in testing if edge[2]==1]

	const_aupr = float(len(positive_edges)) / float(len(testing))
		
	#Increment
	increment = (max_score - min_score) / len(testing)
	
	precision = []
	recall = []
	fprs = []

	
	start = min_score

	while start <= max_score:
		#Extract the edge index above a certain threshold
		indexes = [i for i in range(len(edge_scores)) if edge_scores[i]>=start]
		
		""" Edges above threshold are the correct ones """
		#Extract the edges 
		extracted_edges = [current_network[position]+(1,) for position in indexes]

		""" All the extracted edges are true in nature or they hold the value of 1 """

		#Precision - > How many results / edges are classified as correct
		common = 0 

		for edge in extracted_edges:
			for link in positive_edges:
				if edge[0] == link[0] and edge[1] == link[1]:
					common += 1


		precision.append(float(common) / float(len(extracted_edges)))
		recall.append(float(common)/ float(len(positive_edges)))
		fprs.append(float(len(extracted_edges) - common) / float(len(testing) - len(positive_edges)))

		start += increment


	AUPR_curve = metrics.auc(np.array(recall),np.array(precision))
	AUC_curve = metrics.auc(np.array(fprs),np.array(recall))

	#plt.plot(np.array(fprs),np.array(recall))
	#plt.show()

	#Plots
	#fig, ax = plt.subplots()
	#ax.plot(np.array(fprs),np.array(recall), c='black')
	#line = mlines.Line2D([0, 1], [0, 1], color='red')
	#transform = ax.transAxes
	#line.set_transform(transform)
	#ax.add_line(line)
	#plt.show()

	

	return AUC_curve,AUPR_curve



#Function to extract the predicted scores
def get_scores(predicted_matrix,testing_set,unique_tfs):

	testing_set_scores = []

	actual_labels = []

	for edge in testing_set:
		testing_set_scores.append(predicted_matrix[unique_tfs.index(edge[0])][unique_tfs.index(edge[1])])
		actual_labels.append(edge[2])

	#a,b  = generate_graph(testing_set_scores,testing_set)
	score = metrics.roc_auc_score(actual_labels,testing_set_scores)

	fpr, tpr, thresholds = metrics.roc_curve(actual_labels,testing_set_scores,drop_intermediate=False)

	fig, ax = plt.subplots()
	ax.plot(np.array(fpr),np.array(tpr), c='black')
	line = mlines.Line2D([0, 1], [0, 1], color='red')
	transform = ax.transAxes
	line.set_transform(transform)
	ax.add_line(line)
	plt.show()

	
	print score

	return score

#Main Function 
def main():
	tf, data_matrix = create_data_matrix()

	data_matrix = data_matrix.astype(float)	

	#print mutual_info_score(temp_1,temp_2)	

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

	positive_samples = [link + (1,) for link in positive_interactions]
	negative_samples = [link + (-1,) for link in negative_interactions]

	total_samples = positive_samples + negative_samples

	random.shuffle(total_samples)

	#Compute the matrix
	main_matrix = np.zeros((len(unique_tfs),len(unique_tfs)))

	#Ground Truth Matrix populated with initial scores
	fill_matrix(main_matrix,total_samples,unique_tfs)
	

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

		P, Q = create_factorization_model(training_matrix,4,len(unique_tfs),data_matrix,unique_tfs)

		#Matrix after training
		predicted_matrix = np.matmul(P,Q)

		#Get the scores for the testing set
		auc = get_scores(predicted_matrix,testing_set,unique_tfs)

		
		AUC.append(auc)



	print np.mean(np.array(AUC))
	
	
  
	return




main()


