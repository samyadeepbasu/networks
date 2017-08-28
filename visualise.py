""" Script for visualising the cells according to their expressions in low dimensions """


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
from sklearn.manifold import TSNE, SpectralEmbedding, Isomap
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

	return data_matrix_array


#Function to normalise the state matrix to consolidate all gene expression values from 0 to 1
def normalise(state_matrix):
	state_matrix = state_matrix.astype(float)
	
	#Normalise the matrix 
	matrix = np.array([ (item - np.min(item))/(np.max(item) - np.min(item)) for item in state_matrix])

	return matrix


def reduce_dimensions(matrix):
	#Visualising using t-SNE
	#reduced_matrix = TSNE(n_components=2).fit_transform(matrix)

	reduced_matrix = SpectralEmbedding(n_components=2).fit_transform(matrix)

	#reduced_matrix = Isomap(n_neighbors = 5, n_components=2).fit_transform(matrix)

	X = reduced_matrix[:,0]
	y = reduced_matrix[:,1]

	plt.scatter(X,y)
	plt.show()



def main():
	#Gene Expression Matrix
	data_matrix = create_data_matrix()

	#Transpose the matrix 
	new_data_matrix = data_matrix.transpose()

	#Normalize the gene expression values in each cell
	normalized_matrix = normalise(new_data_matrix)
    
    #Reduce the dimensions to two and three and visualise the cells
	reduce_dimensions(normalized_matrix)



main()
