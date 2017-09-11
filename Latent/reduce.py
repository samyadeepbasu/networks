""" Generation of Latent Representation of Cells by Machine Learning Methods 
	  Basic Intuition : Train a Model to accurately predict the cell labels across given the expression labels  
	  Dataset : Combination of 3 sets, captured during progression of Mouse Cells                                 """



import numpy as np 
import math 
from sklearn.manifold import TSNE, SpectralEmbedding, Isomap
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt

#Function to process the labelled data and separate into training and testing set
def processing():
	#Open the file 
	f = open('training.txt','r')

	#Expression Levels for cells
	dataset = f.readlines()

	#Convert into list -> 402 single cells
	sample_list = dataset[0].split()[1:]
	
	#Labels for the cells
	labels = dataset[1].split()[1:]

	#Weights for the cells
	weights = dataset[2].split()[1:]
	
	#Gene Labels -> 9437 genes
	genes = []
	
	#Expression Levels for each gene
	expressions = []

	for i in range(3,len(dataset)):
		genes.append(dataset[i].split()[0])
		expressions.append(dataset[i].split()[1:])



	return genes, expressions, labels



#Function to create a training set with the cells as vectors and genes as features
def create_training(expressions,cells):
	#Convert from string to float
	expressions = np.array(expressions).astype(float)

	#Transpose the expressions
	expressions = expressions.transpose()

	#Unique labels
	unique_cells = list(set(cells))
	
	#Indexing
	y = np.array([unique_cells.index(cell) for cell in cells])
	

	return unique_cells, expressions, y


#Function to visualise the cells
def visualise_cells(X,Y):
	#t-Stochastic Neighbor Embedding for visualising the cells wrt the labels
	tsne = TSNE(n_components=2).fit_transform(X)
	
	plt.figure("Visualisation of cells")
	plt.scatter(tsne[:,0],tsne[:,1],c=Y*3)
	plt.show()

	return
	

def main():
	#Gene Labels, Expression Measurements (log(TPM) Normalized) , Cell labels
	genes, expressions, cells = processing()

	#Create Training Set
	cell_index, X, Y  = create_training(expressions,cells)

	visualise_cells(X,Y)











main()