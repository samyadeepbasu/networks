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
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor, GradientBoostingRegressor
from sklearn import metrics
import networkx as nx
from numpy import inf
from scipy.stats import pearsonr, spearmanr


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
	return spearmanr(current_target_expression,data_matrix[targets[i]])[0]


#Function to create and initialise the model 
def create_model(training,testing,data_matrix,k):
	
	#Extract only the positive edges from the training 
	training = [edge for edge in training if edge[2]==1]
	
	edge_scores = []

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

		for i in range(len(targets)):
			scores.append(distance(current_target_expression,data_matrix,targets,i))

		if len(scores)<k:
			edge_scores.append(sum(scores))

		else:
			score_index_sort = np.array(list(reversed(np.argsort(scores).tolist())))
			top_k_scores = scores[:k]
			edge_scores.append(sum(top_k_scores))


	
	#Once each edge got a score - Move the threshold and generate a AUPR and AUC
	auc, aupr = generate_graph(edge_scores,testing)

		
	return auc, aupr


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

	

	return AUC_curve,AUPR_curve


def main():
	#Transcriptional Factors along with Data Matrix
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

	positive_samples = [link + (1,) for link in positive_interactions]
	negative_samples = [link + (0,) for link in negative_interactions]

	total_samples = positive_samples + negative_samples
	
	#Mix the positive and negative samples randomly
	#random.shuffle(total_samples)

	#Split into Training and Testing Data
	splitted_sample = split(total_samples)
	
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



		auc, aupr = create_model(training_set,testing_set,data_matrix,10)

		AUC.append(auc)
		AUPR.append(aupr)
	#	const.append(const_aupr)


	print np.mean(np.array(AUC))




main()