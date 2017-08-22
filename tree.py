""" Gene Regulatory Network - Reconstruction : Data Processing and Inference using Tree based method _ Random Forests - For Moignard Dataset  """

#Libraries
import d3py
import math
import openpyxl as px
import numpy as np
import pandas as pd
from autoencoder import AutoEncoder
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples,auc
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import networkx as nx


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


#Function to process and normalize the matrix
def normalize(data_matrix):
	#Mean of the rows
	mean = np.mean(data_matrix,axis=1)

	#Standard Deviation
	standard_deviation = np.std(data_matrix,axis=1)

	for i in range(0,len(data_matrix)):
		data_matrix[i] = (data_matrix[i] - mean[i]) / standard_deviation[i]

	return data_matrix 



#Function to get top 20 variable transcription factors
def get_top_variable(expression_matrix,genes):
	#List for storing the variances
	variances = []

	for i in range(0,len(genes)):
		#Obtain the column for current gene expression
		current_gene_expression = expression_matrix[:,i]

		#Check the variance score and append to the list
		variances.append(np.var(current_gene_expression))


	#Get the Indexes of the Top Variances
	indexes = sorted(range(len(variances)), key=lambda i: variances[i],reverse=True)[:20]

	#Extract the genes with these index
	new_gene_list = [genes[index] for index in indexes]

	#Extract from the Original matrix the needed columns
	new_expression_matrix = expression_matrix[:,indexes]

	return new_expression_matrix, new_gene_list



#Function to reduce the dimensionality of the dataset by using Kernel PCA which is also good for eliminating noise
def reduce_dimensionality(data_matrix):
	#Instantiation : Gamma => Parameter for RBF Kernel, default : 1/n_features
	k_pca = KernelPCA(kernel='rbf',gamma = 10)   

	#Fit the dataset in the model 
	k_pca.fit(data_matrix)

	#Transformed Dataset
	transformed_matrix = k_pca.transform(data_matrix)

	return transformed_matrix

#Function to obtain gene rankings
def get_gene_rankings(data_matrix):
	#List for ranking genes
	ranking_list = []
	#For each gene create a training set along with its ground truth
	for i in range(0,100):
		#Separate the Training Set along with the Ground Truth
		X,y = separate_matrix(data_matrix.copy(),i)

		#Obtain the top regulators after fitting the Regression Tree
		top_regulators_tree = get_top_forest_features(X,y)

		#Obtain the top regulators after fitting the Regression Trees in a Random Forest Setting
		#top_regulators_forest = get_top_forest_features(X,y)

		#Add the top regulators to the main ranking list
		ranking_list.append(top_regulators_tree)


	return ranking_list


#Function to get the most important features corresponding to a particular gene for a Random Forest Setting -- 200 Trees
def get_top_forest_features(X,y):
	#Initializing the Forest Regression - By default bootstrap is true
	forest_regressor = RandomForestRegressor(n_estimators = 200,criterion = 'mse')

	#Fit the training data into the Model
	forest_regressor.fit(X,y)

	#Extract the important features corresponding to a particular gene
	important_features = forest_regressor.feature_importances_

	return important_features


#Function to get the most important features corresponding to a particular gene
def get_top_tree_features(X,y):
	#Initializing the Decision Tree Regressor
	regressor = DecisionTreeRegressor(max_depth= 40)

	#Fit the Training Data into the Model
	regressor.fit(X,y)
	
	#Extract the important features based on the reduction in Gini Importance - This should ideally be replaced by Reduction in Variance
	important_features = regressor.feature_importances_

	return important_features


#Function to separate the matrix and formulate training set
def separate_matrix(data_matrix,i):
	#Ground Truth for the First Gene
	y = data_matrix[:,i]

	#Training Set
	X = np.delete(data_matrix,i,axis=1)

	return X,y

#Function to give the names of the genes as a list
def get_gene_list():
	#Open the file 
	f = open('genes.txt','r')
	
	#Conversion into String
	a = str(f.read())

	#Conversion into List	
	genes = a.split("\t")

	return genes


#Function to find the covariance of Two Random Variables - Parameters : Numpy Arrays
def find_covariance(X,Y):
	#Mean of the X numpy array
	mean_x = np.mean(X)

	#Mean of the Y numpy array
	mean_y = np.mean(Y)

	covariance = 0

	#Number of observations for each random variable
	N = len(X)

	for i in range(0,len(X)):
		covariance += (X[i] - mean_x)*(Y[i] - mean_y) 

	covariance = covariance / N

	return covariance


#Function to obtain the visualization for each gene with respect to its regulators - Individual Visualization
def get_visualization(top_regulator_dict):

	for gene in top_regulator_dict:
		#edges.append((top_regulator_dict[gene][0],gene))
		#edges.append((top_regulator_dict[gene][1],gene))
		#edges.append((top_regulator_dict[gene][2],gene))

		#Create a NetworkX Directed Graph
		graph = nx.DiGraph()
		
		#For storage of connections as tuples in a list
		edges = []
		
		#Populating Edges
		for regulator in top_regulator_dict[gene]:
			edges.append((regulator,gene))
		
		#Addition of Edges into NetworkX instance
		graph.add_edges_from(edges)

		#Draw the Graph
		nx.draw(graph,node_size = 1400,with_labels = True)

		#Save Figure
		plt.savefig("visualizations/" + gene+ ".png",format="PNG")
		plt.clf()

		graph.remove_edges_from(edges)
	
	#For Interactive Visualization
	#with d3py.NetworkXFigure(graph,width=500,height=500) as p:
	#	p += d3py.ForceLayout()
	#	p.show()	

#Function to create the network by moving threshold
def get_results(gene_rankings,tf,reality):

	
	threshold = 0.01

	precision = []

	recall = []
	
	for b in range(0,200):
		top_regulators = []

		for i in range(0,len(tf)):

			tf_temporary = list(tf)

			tf_temporary.remove(tf[i])

			temp_scores = gene_rankings[i]

			#Get the indexes above threshold
			numbers = map((lambda x: np.where(temp_scores == x) if x>threshold else None ),temp_scores)

			#Remove None from the list
			regulator_index = [x[0][0] for x in numbers if x is not None]

			#Make tuples in (target,regulator) format

			topmost_regulators = [(tf[i],tf_temporary[m]) for m in regulator_index]

			top_regulators += topmost_regulators
			

			

		#Create indices for the top most regulators
		top_reg = [(tf.index(pair[0]),tf.index(pair[1])) for pair in top_regulators]

		common = 0 

		for i in top_reg:
			for j in reality:
				if i[0] == j[0] and i[1] == j[1]:
					common += 1


		#print common 
		precision.append(float(common)/ float(len(top_reg)))
		recall.append(float(common)/float(len(reality)))

		threshold += 0.00025

	print precision
	print recall
	print len(precision)
	print len(recall)

	precision = np.array(precision)
	recall = np.array(recall)

	area = auc(recall,precision)
	print "Area:"
	print area
	plt.plot(recall,precision)
	plt.show()


#Function to get the ground truth for the dataset
def ground_truth():
	#Open and Initialise the File
	g_file = open('ground_truth/stamlab_for_data3.txt','r')

	#Conversion of the interactions in appropriate format
	interactions = [ (int(line.split()[2]),int(line.split()[3])) for line in g_file.readlines()]
	
	return interactions


#Function to infer the top potential regulators for each gene
def infer_edges(gene_rankings,genes):
	#Dictionary for storing top regulators for each gene
	top_regulator_dict = {}
	
	#Initialize Dictionary
	for i in range(0,len(genes)):
		top_regulator_dict[genes[i]] = []

	#Infer the top 3 regulators for each gene
	for i in range(0,len(genes)):
		#Temporary List for storing Genes
		a = list(genes)

		#Remove the Gene from the List 
		a.remove(genes[i])

		#Get the ranking list for the gene
		ranking = gene_rankings[i]

		#Obtaining indexes for top three scores
		indexes = sorted(range(len(ranking)), key=lambda i: ranking[i])[-3:]

		#Append the top most regulators
		for index in indexes:
			top_regulator_dict[genes[i]].append(a[index])


	return top_regulator_dict


#Function to Select the top regulators based on covariance - Covariance > 0 : Activation, Covariance<0 : Repression, Covariance = 0 : Neutral in nature
def find_regulators(top_regulator_dict,processed_matrix,gene_list):
	#Top regulators based on covariance 
	top_reg = {}
	neg_reg = {}

	#Initialize Dictionary - It will be appended with top regulators based on covariance
	for gene in gene_list:
		top_reg[gene] = []
		neg_reg[gene] = []

	#For each gene check how the top regulators vary with the output
	for gene in top_regulator_dict:
		#Top Regulators
		top_regulators = top_regulator_dict[gene]

		#Index of the Gene from the Main Gene List
		index = gene_list.index(gene)

		#Extract the column corresponding to the gene
		X = processed_matrix[:,index]

		for regulator in top_regulators:
			#Index for the regulator
			regulator_index = gene_list.index(regulator)

			#Extract the observations associated with the Random Variable
			Y = processed_matrix[:,regulator_index]
			
			#Get the covariance
			cov = find_covariance(X,Y)
			
			#Extract only the top regulators which cause activation
			if cov > 0: 
				top_reg[gene].append(regulator)

			#Extraction of the negative regulators
			if cov < 0:
				neg_reg[gene].append(regulator)


	return top_reg,neg_reg


#Main function to call supporting functions
def main():
	#Cells and the data matrix consisting of Gene Expression Data
	genes, data_matrix = create_data_matrix()

	data_matrix = data_matrix.transpose()

	data_matrix = data_matrix.astype(float)
	
	#Conversion into log values
	#data_matrix = np.log(data_matrix)

	#genes = get_gene_list()

	##new_data_matrix, new_gene_list = get_top_variable(data_matrix, genes)

	#Get a reduced dimensionality matrix
	#transformed_matrix = reduce_dimensionality(data_matrix.transpose())

	#Normalized matrix after reducing to zero mean and unit variance
	processed_matrix = normalize(data_matrix)

	#Gene Ranking for regulation of a specific gene
	gene_rankings = get_gene_rankings(processed_matrix)

	reality = ground_truth()

	get_results(gene_rankings,genes,reality)

	#Get the list of Genes
	#genes = get_gene_list()

	#Get the top ranking potential regulators for a particular gene
	#top_most_regulators = infer_edges(gene_rankings,genes)

	##print top_most_regulators
	
	#Find the top most positive regulators for a particular gene
	#positive_regulators,negative_regulators = find_regulators(top_most_regulators,processed_matrix,genes)

	#print positive_regulators
	#print "#"
	#print negative_regulators

	#get_visualization(positive_regulators)




main()



