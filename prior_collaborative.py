""" Baseline Method for using the Prior Graph to predict edges, when combined with gene expression  """

#############################################################################################################
#############################################################################################################

""" Assumptions : If multiple genes are regulated by a regulator, the targets have some similarity amongst themselves  
		Base Algorithm : 
			1. For each edge in the graph, delete the edge 
			2. Compute Similarity of the deleted target, with that of it's co-targets  
			3. Assign a score to the edge based on similarity                              
			4. Set threshold above which edge = 1, else edge = 0 
			5. Move threshold to get AUPR and AUC curve                                                      
																		   """
#############################################################################################################
#############################################################################################################


#Libraries
import math
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor, GradientBoostingRegressor
from sklearn import metrics
import networkx as nx
from numpy import inf
from scipy.stats import pearsonr


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

#Function to create the local model for computations
def local_model(data_matrix,unique_regulators,interactions):
	#Parameter for finding similarity with 
	k = 4 

	unique_tfs = find_unique_tfs(interactions)
	
	final_weight_list = []

	""" For each edge, remove it and compute the similarity with the other targets with top similarity scores """
	for regulator in unique_regulators:

		#Extract the targets -- Positive Samples
		targets = get_targets(regulator,interactions)

		#Negative Samples
		negative_samples = list(set(unique_tfs) - set(targets))

		negative_samples_temp = list(negative_samples)

		negative_samples_temp.remove(regulator)

		for target in targets:
			#Make a copy of the targets
			temp = list(targets)
			
			#Remove the current edge from the network
			temp.remove(target)

			if len(temp) == 0:
				#If there is only one edge for the particular regulator --> Initialise the matrix as -1
				final_weight_list.append([(regulator,target),-1])
			
			#If the number of neighbours are less than k, compute similarity with all the neighbours
			elif len(temp) < k :

				#Compute Pearson Correlation with each of the neighbours
				neighbour_expression = data_matrix[:][temp]

				target_expression = data_matrix[target]

				sim_scores = [pearsonr(expression,target_expression)[0] for expression in neighbour_expression]

				final_weight_list.append([(regulator,target),sum(sim_scores)])



			else: #Condition where there are more than or equal to k neighbours
				neighbour_expression = data_matrix[:][temp]

				target_expression = data_matrix[target]
				
				#Pearson Correlation
				sim_scores = np.array([pearsonr(expression,target_expression)[0] for expression in neighbour_expression])

				#Extract the Indexes of Top few scores
				ordered_indexes = np.flip(np.argsort(sim_scores),axis=0)

				#Top Scores
				top_k = ordered_indexes[:k]

				similarity = np.sum(sim_scores[:][top_k])
				
				final_weight_list.append([(regulator,target),similarity])

			
			#print len(neighbours)
			#print neighbours

		""" According to the Algorithm the non_edges should have lesser similarity with the existing edges """

		for node in negative_samples_temp:
			#If the number of neighbours are less than k, compute similarity with all the neighbours
			if len(targets) < k :

				#Compute Pearson Correlation with each of the neighbours
				neighbour_expression = data_matrix[:][targets]

				target_expression = data_matrix[node]

				sim_scores = [pearsonr(expression,target_expression)[0] for expression in neighbour_expression]

				final_weight_list.append([(regulator,node),sum(sim_scores)])



			else: #Condition where there are more than or equal to k neighbours
				neighbour_expression = data_matrix[:][targets]

				target_expression = data_matrix[node]
				
				#Pearson Correlation
				sim_scores = np.array([pearsonr(expression,target_expression)[0] for expression in neighbour_expression])

				#Extract the Indexes of Top few scores
				ordered_indexes = np.flip(np.argsort(sim_scores),axis=0)

				#Top Scores
				top_k = ordered_indexes[:k]

				similarity = np.sum(sim_scores[:][top_k])
				
				final_weight_list.append([(regulator,node),similarity])

		

	return final_weight_list
	

def main():
	#Transcriptional Factors along with Data Matrix
	tf, data_matrix = create_data_matrix()

	data_matrix = data_matrix.astype(float)

	#Ground Truth : (Regulator, Target)
	interactions = ground_truth()
	
	#Regulator List
	regulators = [item[0] for item in interactions]
	
	#Generation of unique Regulators for Local Model
	unique_regulators = list(set(regulators))

	#Edge importances based on local neighbourhood
	edge_importances = local_model(data_matrix,unique_regulators,interactions)

	







	


main()