""" Imputation of Dropouts based on averaging out expression levels from Similar Cells  """


#Libraries
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from numpy import inf

""" Algorithm : 
                1. Cluster Cells by using an ensemble of methods 
                2. For each method compute the averaging instance to replace the dropout point
                3. Average out all the dropout point results from different clusterings to generate a singular dropout score

                                                                        """

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

	return data_matrix_array


def perform_imputation(data_matrix):
	#
	return

def main():
	data_matrix = create_data_matrix()

	#Transpose the data matrix
	data_matrix = data_matrix.transpose()

	#Perform imputation onto the matrix
	imputed_matrix = perform_imputation(data_matrix)





main()
