""" Gene Regulatory Inference using Lasso based Feature Selection and Stability Selection  """

#Libraries
import d3py
import math
import openpyxl as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import KMeans
import networkx as nx
from sklearn.linear_model import Lars
import random


#Function to import data from Excel File
def import_data(file_name):
    #Creating a pd instance
    x1 = pd.ExcelFile(file_name)

    #Parsing the necessary sheet
    df = x1.parse('Raw Ct values')

    #df = x1.parse('dCt_values.txt') 

    #Population of Cells in the Experiment
    a = df['Cell'].as_matrix()

    #Removal of Redundant Column
    df = df.drop('Cell',axis=1)

    #Conversion into numpy matrix
    b = df.as_matrix()

    return a,b

#Function to process and normalize the matrix
def normalize(data_matrix):
	#Mean of the rows
	mean = np.mean(data_matrix,axis=1)

	#Standard Deviation
	standard_deviation = np.std(data_matrix,axis=1)

	for i in range(0,len(data_matrix)):
		data_matrix[i] = (data_matrix[i] - mean[i]) #/ standard_deviation[i]

	return data_matrix 


#Function to give the names of the genes as a list
def get_gene_list():
	#Open the file 
	f = open('genes.txt','r')
    
    #Conversion into String
	a = str(f.read())

	#Conversion into List	
	genes = a.split("\t")

	return genes


#Function to get the ranks for each regulator gene
def get_ranks(coefficients,gene_copy):
	#Ranks
	ranks = []

	#Final Rank Score
	rank_score = {}
	
	for coeff in coefficients:
		#Extract the indexes of the top genes
		indexes = sorted(range(len(coeff)), key=lambda i: coeff[i],reverse=True)[-len(gene_copy):]
		
		#Append to main rank list
		ranks.append(indexes)


	for i in range(0,len(gene_copy)):
		#Temporary variable for summation of ranks
		total_rank = 0

		for rank in ranks:
			total_rank += rank[i]
        
        #Effective Rank after all the iterations
		effective_rank = total_rank / len(ranks)
        
        #Append to Dictionary
		rank_score[gene_copy[i]] = effective_rank


	return rank_score


#Function to perform Stability Selection : R=> Number of iteration to run, matrix has P-1 entries, genes after removal
def stability_selection(expression_matrix,genes,R,y,gene_copy):
	#Final Score for each of the transcription factors
	score = []

	#Coefficients for each iteration
	coefficients = []

	#Run the Selection Algorithm for R/2 times
	for i in range(0,R/2):
		#Indexes for Randomly splitting the data into equal halves
		indices = range(0,len(genes)-1)

		#Randomly Shuffle the indices
		random.shuffle(indices)

		#Split into two parts
		first_half = indices[:len(genes)/2]
		second_half = indices[len(genes)/2:]

		#First Half of the Expression Matrix
		extract_first_half = expression_matrix[:,first_half]

		#Second Half of the Expression Matrix
		extract_second_half = expression_matrix[:,second_half]

		#Randomly Perturb Data by multiplying the expression of candidate TF's with a number b/w (alpha,1), where alpha belongs to (0,1)
		alpha = 0.19

		#Perturbation
		perturbation = random.uniform(alpha,1)

		#Multiply the expression matrix
		perturbed_first_half = extract_first_half * perturbation

		perturbed_second_half = extract_second_half * perturbation

		#Run LARS on each of them to get the score
		coeff = Lars()

		#Fit the First Half
		coeff.fit(perturbed_first_half,y)
        
        #Result for the first half of the split
		result_first_half = coeff.coef_

		#Fit the second half
		coeff.fit(perturbed_second_half,y)

		#Result for the second half of the split
		result_second_half = coeff.coef_

		temp_dict = {}

		#Creation of Singular Score Array
		for i in range(0,len(first_half)):
			temp_dict[first_half[i]] = result_first_half[i]


		for i in range(0,len(second_half)):
			temp_dict[second_half[i]] = result_second_half[i]


        #Append the values into the empty list
		coeff_list = []

		for val in temp_dict.values():
			coeff_list.append(val)

		#Append to main coeff list
		coefficients.append(coeff_list)

	
    #Ranks for Each Regulator Gene
	ranks = get_ranks(coefficients,gene_copy)

	return ranks

#Function to extract the top regulators for a Particular Gene
def find_top_regulators(rank_dict):
	#Top Regulators
	return sorted(rank_dict.items(),key = lambda x:-x[1],reverse=True)[:4]


#Function to perform Least Angle Regression : For each gene obtain the coefficients of the regression equation
def perform_LARS(normalized_matrix,genes):
	#Number of Genes
	no_genes = len(genes)

	#Dictionary for top regulators for each gene
	regulators = {}
    
	for i in range(0,no_genes):
		#Current Gene for which the Top Regulators are being found
		current_y = normalized_matrix[:,i]

		#Create a copy of the matrix
		temp_matrix = normalized_matrix.copy()

		#Remove the current feature
		temp_matrix = np.delete(temp_matrix,i,axis=1)		

		#Computation of the coefficients after training with Least Angle Regression Method
		coefficients = Lars()

		#Fit the Model
		coefficients.fit(temp_matrix,current_y)

		#Coefficient values
		coeff_values = coefficients.coef_

		#Copy the genes into a temporary list
		gene_copy = list(genes)

		#Remove the Gene to create the appropriate indexes
		gene_copy.remove(genes[i])
        
        #Perform Stability Selection to get an effective rank of the top regulators
		rank_dict_score = stability_selection(temp_matrix,genes,2000,current_y,gene_copy)

		#Top Regulators
		top_regulators = find_top_regulators(rank_dict_score)

		#Append to regulators
		regulators[genes[i]] = top_regulators	


	return regulators
	


#Main Function
def main():
	#Input Gene Expression Data Matrix
	cells, data_matrix = import_data('data.xlsx')

	#Conversion into Log values
	#data_matrix = np.log(data_matrix)

	#Normalize the matrix to zero mean and unit variance
	normalized_matrix = normalize(data_matrix)

	#Get the list of Genes
	genes = get_gene_list()

	#Perform Least Angle Regression to get the coefficients
	top_regulators = perform_LARS(normalized_matrix,genes)

	for key,value in top_regulators.items():
		print key
		print value
		print "#"
	






main()