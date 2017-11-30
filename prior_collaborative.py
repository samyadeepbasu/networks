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


#Function to calculate the similarity metric
def find_pearson_similarity(expression,target_expression):

	return pearsonr(expression,target_expression)[0]


#Function to create the local model for computations
def local_model(data_matrix,unique_regulators,interactions,k):
	#Parameter for finding similarity with neighbourhood target genes	

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

				sim_scores = [find_pearson_similarity(expression,target_expression) for expression in neighbour_expression]

				final_weight_list.append([(regulator,target),sum(sim_scores)])



			else: #Condition where there are more than or equal to k neighbours
				neighbour_expression = data_matrix[:][temp]

				target_expression = data_matrix[target]
				
				#Pearson Correlation
				sim_scores = np.array([find_pearson_similarity(expression,target_expression) for expression in neighbour_expression])

				#Extract the Indexes of Top few scores
				ordered_indexes = np.argsort(sim_scores).tolist()

				ordered_indexes = list(reversed(ordered_indexes))

				ordered_indexes = np.array(ordered_indexes)

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

				sim_scores = [find_pearson_similarity(expression,target_expression) for expression in neighbour_expression]

				final_weight_list.append([(regulator,node),sum(sim_scores)])



			else: #Condition where there are more than or equal to k neighbours
				neighbour_expression = data_matrix[:][targets]

				target_expression = data_matrix[node]
				
				#Pearson Correlation
				sim_scores = np.array([find_pearson_similarity(expression,target_expression) for expression in neighbour_expression])

				#Extract the Indexes of Top few scores
				ordered_indexes = np.argsort(sim_scores).tolist()

				ordered_indexes = list(reversed(ordered_indexes))

				ordered_indexes = np.array(ordered_indexes)

				#Top Scores
				top_k = ordered_indexes[:k]

				similarity = np.sum(sim_scores[:][top_k])
				
				final_weight_list.append([(regulator,node),similarity])

		

	return final_weight_list

#Function to generate the AUC and AUPR scores
def get_graph(edge_importances,interactions):
	#Get minimum and maximum value
	edge_score = [edge[1] for edge in edge_importances]

	edge_score = sorted(edge_score)

	#Minimum and Maximum
	minimum = min(edge_score)

	maximum = max(edge_score)

	increment = (maximum - minimum) / len(edge_importances) 

	threshold = minimum 

	#How many amongst those extracted are correct 
	precision = []

	#How many correct from the ground truth has been extracted
	recall = []

	fpr = []

	while threshold<maximum:
		#Extract the edges above the threshold
		
		extracted_edges = [edge[0] for edge in edge_importances if edge[1]>threshold]

		common = 0
		for edge in extracted_edges:
			for link in interactions:
				if edge[0] == link[0] and edge[1] == link[1]:
					common += 1
		


		precision.append(float(common) / float(len(extracted_edges)))
		recall.append(float(common) / float(len(interactions)))
		fpr.append((float(len(extracted_edges)) - float(common)) / float(1560 - len(interactions)))

		
		
		if float(common) / float(len(extracted_edges)) > 0.6 and float(common) / float(len(extracted_edges)) < 0.61:
			

			plot_graph = []

			for edge in extracted_edges:
				flag = 0
				for link in interactions:
					if edge[0] == link[0] and edge[1] == link[1]:
						#Common element found 
						plot_graph.append((edge[0],edge[1],1))
						flag = 1

				if flag == 0:
					plot_graph.append((edge[0],edge[1],0.5))


			g = nx.MultiGraph()

			g.add_weighted_edges_from(plot_graph)

	

			edgewidth = [ d['weight'] for (u,v,d) in g.edges(data=True)]
			

			#pos = nx.spring_layout(g, iterations=50)
			pos = nx.random_layout(g)
			nx.draw_networkx_edges(g,pos,edge_color = edgewidth)
			plt.show()

		#Increment Step
		threshold += increment*4
	

	#Calculation for area 
	AUC_curve = metrics.auc(np.array(fpr),np.array(recall))
	#print AUC_curve
	AUPR_curve = metrics.auc(np.array(recall),np.array(precision))

	plt.plot(fpr,recall)
	plt.xlabel('False Positive Rate')
	plt.ylabel('Recall')
	plt.show()

	plt.plot(recall,precision)
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.show()	

	

	return AUC_curve, AUPR_curve 




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
	
	AUC_list = []

	#for k in range(2,20):

	k = 10

	#Edge importances based on local neighbourhood
	edge_importances = local_model(data_matrix,unique_regulators,interactions,k)

	#Calculate AUPR and AUC by moving the threshold
	get_graph(edge_importances,interactions)



main()