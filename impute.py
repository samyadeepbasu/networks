""" Imputation of Dropouts based on averaging out expression levels from Similar Cells  """


#Libraries
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from numpy import inf
from scipy.stats import pearsonr
from sklearn.decomposition import PCA,KernelPCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE, SpectralEmbedding, Isomap
from numpy import inf

""" Algorithm : 
				1. Cluster Cells by using an ensemble of Clustering Algorithms for variable number of clusters 
				2. For each method compute the averaging instance to replace the dropout point
				3. Average out all the dropout point results from different clusterings to generate a singular dropout score

																																 """

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

	return data_matrix_array


#Function to replace dropouts in the same cluster
def replace_dropouts(clusters):
	#Replaced Data Matrix
	replaced_matrix = []
	total = []
	
	for cluster in clusters:
		#Temporary Matrix
		temp_cluster = clusters[cluster]

		temp = []
		
		#Replace the dropouts with the average expression values for each gene
		for gene in temp_cluster:
			#Indexes which have to be replaced
			indexes = np.where(gene == 0)

			#Mean 
			mean = np.mean(gene)

			#Replace Values
			gene[:][indexes[0]] = mean

			temp.append(gene.tolist())


		temp = np.array(temp)
		
		temp = temp.transpose()

		replaced_matrix += temp.tolist()	
	

	replaced_matrix = np.array(replaced_matrix)


	return replaced_matrix


#Function to calculate the similarity matrix
def similarity(data_matrix):
	#Similarity Matrix 
	similarity_matrix = []

	#Compute Similarity Matrix with Pearson Correlation
	for i in range(0,len(data_matrix)):
		temp = []
		for j in range(0,len(data_matrix)):
			temp.append(pearsonr(data_matrix[i],data_matrix[j])[0])

		similarity_matrix.append(temp)


	return similarity_matrix


#Function to perform imputation
def perform_imputation(data_matrix):
	#Cast into float
	data_matrix = data_matrix.astype(float)

	similarity_matrix = similarity(data_matrix)	

	#Number of components ~ 5% of the total cells
	components = int(0.05 * len(similarity_matrix))

	#Reduce Dimensions
	d = KernelPCA(n_components=2)#components)

	reduced_data = d.fit_transform(similarity_matrix)
	
	k_start = 10
	k_end = 15

	#Cluster using K-means -- This has to be put inside a loop
	clusterer = KMeans(n_clusters = 3,init='k-means++')

	labels = clusterer.fit_predict(reduced_data)

	#visualise(reduced_data,labels)

	#Create Clusters
	clusters = {}

	#Get unique values
	unique_value = np.unique(labels)
	
	
	for cluster in unique_value:
		#Get indexes
		indexes = np.where(labels==cluster)

		clusters[cluster] = data_matrix[:][indexes[0]].transpose()

		#print data_matrix[:][indexes[0]].transpose().shape

	
	#Replace Dropouts
	new_matrix = replace_dropouts(clusters)


	return new_matrix, reduced_data,labels
	
	

#Normalise the expression levels in the cell -- Convert into Normal Distribution
def normalise(matrix):
	mew = np.mean(matrix,axis=1)

	std = np.std(matrix,axis=1)

	for i in range(0,len(matrix)):
		matrix[i] = (matrix[i] - mew[i]) / std[i]


	return matrix


#Function to visualise the progression of cells
def visualise(imputed_matrix,labels):
	#Normalise before clustering
	#normalised_matrix = normalise(imputed_matrix)
	#normalised_matrix = imputed_matrix

	#Visualise
	#reduced_matrix = SpectralEmbedding(n_components=2).fit_transform(normalised_matrix)
	
	#Visualise 
	#reduced_matrix = KernelPCA(n_components=2).fit_transform(normalised_matrix)
	reduced_matrix = imputed_matrix
	X = reduced_matrix[:,0]
	Y = reduced_matrix[:,1]

	plt.figure("Progression after imputation")

	plt.scatter(X,Y,c=labels)
	plt.show()	

	return

#Function to print the imputed matrix to file 
def print_to_file(imputed_matrix):
	total = 0
	
	#print imputed_matrix[0]

	return

#Main Function
def main():
	data_matrix = create_data_matrix()

	#Transpose the data matrix
	data_matrix = data_matrix.transpose()
	
	#Convert from string to float
	data_matrix = data_matrix.astype(float)

	#data_matrix = np.log(data_matrix)

	#data_matrix[data_matrix == -inf] = 0
	
	#Normalise
	#data_matrix = normalise(data_matrix)

	#Perform imputation onto the matrix
	imputed_matrix ,reduced_data, labels = perform_imputation(data_matrix)

	#Transpose and print to File
	imputed_matrix = imputed_matrix.transpose()

	#Print the Imputed Matrix into a data file for Pseudo Time Measurement
	print_to_file(imputed_matrix)

	
	









	









main()
