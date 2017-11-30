""" Application of Discrete Linear Kalman Filters on Pseudo Time Series Data to generate new measurements for passing onto main Prediction Pipeline   
	Treatment of the expression levels as a Gaussian -> Not 100 percent true                                                            """


from noise_reduction import matrix_pass,normalise
import numpy as np
import matplotlib.pyplot as plt 

""" 
The Discrete Kalman Filters consist of Two Major Steps : 
 1. Prediction from Previous Step
 2. Update to the State with Respect to Noisy Measurements
																  """

#Function to check the distributions
def gaussian(time_array):
	#Mean
	mu = np.mean(time_array)

	#Sigma
	sigma = np.std(time_array)
	
	#Gaussian Distribution Function
	y = np.exp(-np.power(time_array - mu,2.) / 2 * np.power(sigma,2.))    

	plt.figure()

	plt.scatter(time_array,y)

	plt.show()                


#Function to apply the functionalities of the filter
def apply_filter(time_matrix):
	#New Measurements after filtering measurement noise
	updated_measurements = []

	#Previous Updated State -- Assumption : First state measurements are correct
	previous_updated_state = 0  #time_matrix[0].copy()

	#Covariance Matrix for the first state
	previous_covariance_matrix = 1.0

	#Process Variance
	Q = 1e-5
	
	#Estimate of Measurement Variance
	R = 0.1**1

	measure = np.random.normal(-0.3777,0.6,size=100)

	new_state = []

	#Iterate and over the measurements to get a new matrix
	for i in range(1,len(measure)):
		#Getting the measurements of the current state -- This measurement value will be used to get a new estimate
		current_measurement =  measure[i]#time_matrix[i].copy()

		""" Predictions """ 

		#New Predictions from the previous updated state without measurement tuning
		current_prediction = previous_updated_state 
		current_covariance_matrix = previous_covariance_matrix + Q  #Add the process Variance

		""" Tuning with respect to Observed Noisy Measurements """ 

		#Obtain the Kalman Gain
		k = current_covariance_matrix / (current_covariance_matrix + R)

		#Update the new prediction with respect to measurement
		updated_state = current_prediction + k * (current_measurement - current_prediction)
		updated_covariance_matrix = (1 - k) * current_covariance_matrix


		#Initialisation for the next state
		previous_updated_state = updated_state
		previous_covariance_matrix = updated_covariance_matrix

		new_state.append(updated_state)
		print current_measurement
		print "#"

	#print len(new_state)

	temp = range(1,100)

	plt.rcParams['figure.figsize'] = (10, 8)
	plt.figure()
	plt.plot(new_state)
	plt.plot(measure[1:])
	plt.axhline(-0.3777,color='g',label='truth value')
	plt.show()

		
#Function to create bins for the dataset and check if the distribution is normal in nature
def binning(matrix):
	new_matrix = map((lambda x: round(x,1)),matrix)

	#Get the Unique Elements
	unique_elements = list(set(new_matrix))

	count_unique = map((lambda x: new_matrix.count(x)),unique_elements)

	

	plt.figure()
	plt.rcParams['figure.figsize'] = (10, 8)	
	plt.scatter(np.array(unique_elements),np.array(count_unique))
	#plt.plot(np.array(count_unique))
	plt.show()
	

#Function to normalize the dataset by standardisation -- Centre the Data
def standardise(data_matrix):
	#Mean of the set
	means = np.mean(data_matrix,axis=1)

	#Standard Deviation of the Set
	std = np.std(data_matrix,axis=1)

	for i in range(len(means)):
		data_matrix[i] = (data_matrix[i] - means[i]) / std[i]

	
	return data_matrix



#Function to Implement Kalman Filter for Noise Reduction
def kalman_filter():
	#Obtain the Pseudo Time Ordered Data Matrix
	data_matrix, ordered_cells = matrix_pass()

	#Transpose the Matrix to treat each cell as a state
	state_matrix = data_matrix.transpose()

	#Normalise the expression levels across each state
	#normalized_matrix = normalise(state_matrix)
	normalized_matrix = standardise(state_matrix)

	binning(normalized_matrix[0])

	#Obtain the new measurements after applying the Kalman Filter
	new_measurements = apply_filter(normalized_matrix)

	#binning(normalized_matrix[0])




kalman_filter()