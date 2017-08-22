""" Gene Regulatory Network - Reconstruction : Data Processing and Inference using Mutual Information without post processing - For Moignard Dataset  """

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
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import networkx as nx


#Function to import data from Excel File
def import_data(file_name):
    #Creating a pd instance
    x1 = pd.ExcelFile(file_name)

    #Parsing the necessary sheet
    #df = x1.parse('Raw Ct values')

    df = x1.parse('dCt_values.txt') 

    #Population of Cells in the Experiment
    a = df['Cell'].as_matrix()

    #Removal of Redundant Column
    df = df.drop('Cell',axis=1)

    #Conversion into numpy matrix
    b = df.as_matrix()

    return a,b

#Function to reduce the dimensionality of the dataset by using Kernel PCA which is also good for eliminating noise
def reduce_dimensionality(data_matrix):
    #Instantiation : Gamma => Parameter for RBF Kernel, default : 1/n_features
    k_pca = KernelPCA(kernel='rbf',gamma = 10)   

    #Fit the dataset in the model 
    k_pca.fit(data_matrix)

    #Transformed Dataset
    transformed_matrix = k_pca.transform(data_matrix)

    return transformed_matrix



#Function to construct mutual information matrix
def get_mutual_information(data_matrix,genes):
    #Dictionary for storing mutual information for each gene wrt others
    mutual_info = {}

    #Matrix
    gene_matrix = []

    #For each gene find its mutual information with the other genes
    for i in range(0,len(genes)):
        #Initialize Dictionary
        mutual_info[genes[i]] = []

        temp = []

        #Extract the array for the corresponding gene
        X = data_matrix[:,i]

        for j in range(0,len(genes)):
            #Extract the other gene vectors
            Y = data_matrix[:,j]

            #Calculate the score
            mutual_info[genes[i]].append(mutual_info_score(X,Y))
            temp.append(mutual_info_score(X,Y))
            #print mutual_info_regression(np.array(X),np.array(Y))
        
        #Append it to the Gene Matrix
        gene_matrix.append(temp)


    #Conversion into numpy matrix
    gene_matrix = np.array(gene_matrix)

    return gene_matrix


#Function to extract the meaningful edges
def get_top_edges(gene_matrix,genes):
    #Dictionary for storing the connections
    top_connections = {}

    #For each gene obtain the most relevant connections
    for i in range(0,len(genes)):
        #Initialize the dictionary
        top_connections[genes[i]] = []

        #Obtaining indexes for top four scores
        indexes = sorted(range(len(gene_matrix[i])), key=lambda j: gene_matrix[i][j])[-4:]

        top_connections[genes[i]].append(np.array(genes)[indexes])


    for item in top_connections:
        top_connections[item] = (top_connections[item][0]).tolist()


    return top_connections


#Function to Eliminate Edges corresponding to self loops
def eliminate_edges(edges,genes):
    for item in edges:
        if item in edges[item]:
            edges[item].remove(item)


    return edges


#Function to give the names of the genes as a list
def get_gene_list():
    #Open the file 
    f = open('genes.txt','r')
    
    #Conversion into String
    a = str(f.read())

    #Conversion into List   
    genes = a.split("\t")

    return genes


#Function to create a discretized matrix based on Gene Expression Fluctuation Levels
def get_discrete_matrix(data_matrix):
    #New Data Matrix
    new_matrix = []

    #Number of iterations
    iterations = len(data_matrix[0])

    for gene in data_matrix:
        #Temporary List for storage of states : 5 states in total
        temp = []

        for j in range(1,iterations-1):
            #Selection of states based on fluctuation levels
            if gene[j] > gene[j-1] and gene[j+1] > gene[j]:
                temp.append(0)
            elif gene[j] > gene[j-1] and gene[j+1] < gene[j]:
                temp.append(1)
            elif gene[j] < gene[j-1] and gene[j+1] > gene[j]:
                temp.append(2)
            elif gene[j] < gene[j-1] and gene[j+1] < gene[j]:
                temp.append(3)
            else:
                temp.append(4)

        new_matrix.append(temp)

    return np.array(new_matrix)



#Function to call rest of the functions
def main():
    #Cells and the data matrix consisting of Gene Expression Data
    cells, data_matrix = import_data('data.xlsx')

    #Get the list of Genes
    genes = get_gene_list()

    #Transformation of the initial matrix by Kernel PCA - Removal of redundant features and noise (Kernel PCA is robust to noise)
    #transformed_matrix = reduce_dimensionality(data_matrix.transpose())

    #Discretize the matrix 
    discretized_matrix = get_discrete_matrix(transformed_matrix)

    #print len(transformed_matrix)

    #Transpose the matrix to get to the initial format
    #new_data_matrix = transformed_matrix.transpose()

    #Function call to get the matrix formed due to mutual information
    gene_matrix = get_mutual_information(data_matrix,genes)

    

    print gene_matrix[0]
    print gene_matrix[1]

    #Selection of connections from the Gene Matrix to obtain the best connections
    #edges = get_top_edges(gene_matrix,genes)
    
    #Edges after Elimination of the self loops which are prevelant in most of the cases
    #final_edges = eliminate_edges(edges,genes)

    #print len(gene_matrix)


    




main()