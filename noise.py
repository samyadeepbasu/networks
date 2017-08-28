""" Script to check if the Gene Expression Time Series is a White Noise  
     :If White Noise : All the variables are independent and random which renders modelling to be difficult                """


import numpy as np 


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


#Function to order cells by pseudo time
def pseudo_time(data_matrix):
	#Open the file corresponding to Pseudo Time Measurement
	time_file = open('data2/time.txt','r')
	
	#Extraction of Pseudo Time from the List
	ordered_cells = [line.split()[1] for line in time_file.readlines()]


	#Convert into Numpy Array
	ordered_cells = np.array(ordered_cells)

	#Conversion from string to float
	new_ordered_cells = [float(item) for item in ordered_cells]
	
	#Get the Indexes for the sorted order
	indexes = np.argsort(new_ordered_cells)

	#Order the data_matrix
	new_matrix = data_matrix[:,indexes]

	#Convert every expression level in the matrix into floating point number
	new_matrix = new_matrix.astype(float)	
	
	return new_matrix


#Main Function 
def main():
	data_matrix = create_data_matrix()

	
