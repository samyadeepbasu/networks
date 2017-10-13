""" Baseline Method for using the Prior Graph to predict edges, when combined with gene expression  """

#############################################################################################################
#############################################################################################################

""" Assumptions : If multiple genes are regulated by a regulator, the targets have some similarity amongst themselves  
		Base Algorithm : 
			1. Obtain a reduced form of the prior Graph, by deleting a certain percentage of edges -> Training Data
			2. Treat the deleted edges as testing set 
			3. For each edge in the testing set, score the edge according to the scheme 
			  Scoring scheme : 
				-> For the edge in the test set, find the regulator and it's corresponding targets 
				-> Find the similarity of the target in the test edge with that of the remaining targets in the training targets
				-> Churn out the scores corresponding to the top k similar neighbours

			Ideally : 
				-> A potential edge will have a high score and a negative edge should have a low score 

			For each testing fold generate an AUPR curve and average it out overall the folds to generate the final curve                                                 
																															  """
#############################################################################################################
#############################################################################################################


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
import networkx as nx
from numpy import inf
from scipy.stats import pearsonr, spearmanr
from scipy import spatial
from sklearn.metrics import mutual_info_score


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
	#print normalised_time


	return normalised_time


#Function to get the ground truth for the dataset
def ground_truth():
	#Open and Initialise the File
	g_file = open('ground_truth/stamlab_for_data3.txt','r')

	#Conversion of the interactions in appropriate format  -- (Regulator --->  Target)
	interactions = [ (int(line.split()[3]),int(line.split()[2])) for line in g_file.readlines()]

	for item in interactions: #Remove Self-Loops
		if item[0] == item[1]:
			interactions.remove(item)
	
	return interactions

#Function to extract the targets
def get_targets(regulator,interactions):
	targets = []

	for edge in interactions:
		if edge[0] == regulator:
			targets.append(edge[1])

	return targets

#Function to find the unique transcriptional factors 
def find_unique_tfs(interactions):
	temp = []

	for interaction in interactions:
		temp.append(interaction[0])
		temp.append(interaction[1])

	temp = list(set(temp))

	return temp



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


#Function to find the distance
def distance(current_target_expression,data_matrix,targets,i):
	#return spearmanr(current_target_expression,data_matrix[targets[i]])[0]
	#return spatial.distance.correlation(current_target_expression,data_matrix[targets[i]])
	#return mutual_info_score(current_target_expression,data_matrix[targets[i]])
	return rbf_kernel(current_target_expression.reshape(1,-1),data_matrix[targets[i]].reshape(1,-1))
	


#Function to find the distance on similarity between signals
def DTW_distance(current_target_expression,data_matrix,targets,i):
	#To be filled up
	return


#Function to create and initialise the model 
def create_model(training,testing,data_matrix,k):
	
	#Extract only the positive edges from the training 
	training = [edge for edge in training if edge[2]==1]
	
	edge_scores = []
	check = 0
	#Score each edge in the testing data using the training set Graph
	for edge in testing:
		#Get the score for the edge
		regulator = edge[0]
		current_target = edge[1]

		targets = []
		#Get the corresponding genes in the local neighbourhood of edge[1] which is the target
		for link in training:
			if link[0] == regulator:
				targets.append(link[1])

		

		#Score the edge with respect to the present training graph
		current_target_expression = data_matrix[current_target]

		scores = []
		#print check
		
		for i in range(len(targets)):
			scores.append(distance(current_target_expression,data_matrix,targets,i))
			#scores.append(1)

		if len(scores)<k:
			if len(scores) == 0:
				edge_scores.append(0)
			else:
				edge_scores.append(float(sum(scores))/len(scores))
			#edge_scores.append(1)

		else:
			score_index_sort = np.array(list(reversed(np.argsort(scores).tolist())))
			top_k_scores = scores[:k]
			edge_scores.append(float(sum(top_k_scores))/k)


		#print scores
	#Once each edge got a score - Move the threshold and generate a AUPR and AUC
	auc, aupr,const_aupr = generate_graph(edge_scores,testing)

		
	return auc, aupr,const_aupr



#Function to put the different states into bins
def binning(state_matrix, times, k): #Time Passed is sorted

	#temp_state_matrix = state_matrix.copy()

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

	plt.plot(np.array(fprs),np.array(recall))
	plt.show()

	

	return AUC_curve,AUPR_curve,const_aupr


def main():
	#Transcriptional Factors along with Data Matrix
	tf, data_matrix = create_data_matrix()

	data_matrix = data_matrix.astype(float)

	data_matrix = pseudo_time(data_matrix)

	time_series = time()

	state_range = range(300,700)

	main_matrix = data_matrix.copy()

	AUC_total = []

	for k in state_range:

		data_matrix, time_ordered = binning(main_matrix.transpose(),time_series,200)

		data_matrix = data_matrix.transpose()	

		#Ground Truth : Positive Interactions : (Regulator, Target)
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



			auc, aupr,const_aupr = create_model(training_set,testing_set,main_matrix,16)

			AUC.append(auc)
			AUPR.append(aupr)
			#print auc
			#print aupr

			#print const_aupr
			#print "#"




		print np.mean(np.array(AUC))
		print ""
		AUC_total.append(np.mean(np.array(AUC)))
		break



	#X = range(0,len(AUC_total))

	#plt.plot(X,AUC_total)
	#plt.show()
		





main()