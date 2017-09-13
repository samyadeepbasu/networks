"""  Reducing Dimensions -> 100 Genes X 373 Cells """


import numpy as np 
import matplotlib.pyplot as plt 
import math 
import tensorflow as tf
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, SpectralEmbedding,TSNE

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


#Visualise the clusters
def visualise(data_matrix):
	#Reduce Dimensions using PCA
	pca = TSNE(n_components=2).fit_transform(data_matrix)
	X = pca[:,0]
	Y = pca[:,1]

	plt.scatter(X,Y)
	plt.show()


	return

#Convert Data into Gaussian Distribution before passing to AutoEncoder
def gaussian(data_matrix):
	for i in range(len(data_matrix)):
		data_matrix[i] = (data_matrix[i] - np.mean(data_matrix)) / np.std(data_matrix)
		#data_matrix[i] = (data_matrix[i] - max(data_matrix[i])) / (max(data_matrix[i]) - min(data_matrix[i]))

	return data_matrix


#Calculate KullBack Leibler Divergence => Parameters : p_hat (Predicted value), p(Ideal Value) 
def KL_divergence(p_hat,p):
	#Summation of all the activations of the training examples in the batch
	#p_hat = tf.reduce_mean(p_hat,axis=0)

	KL_penalty = tf.reduce_mean(p*tf.log(p) - p*tf.log(p_hat) + (1-p)*tf.log(1-p) - (1-p)*tf.log(1-p_hat))

	return KL_penalty


#Function to train an autoencoder and generate a latent representation of the cells
def train(data_matrix,k):
	#Input without noise
	input_data = data_matrix

	#Output data
	output_data = input_data

	input_data = input_data + 0.2 * np.random.random_sample((input_data.shape))

	#Number of Neurons in the Hidden Unit
	n_hidden = k 

	#Number of Samples
	n_samples = input_data.shape[0]

	#Number of Features / Number of Genes in the input layer
	n_features = input_data.shape[1]

	#Declaring the Tensorflow Variables - None => Can have any number of rows => (none X 100)
	X = tf.placeholder(tf.float32,[None,n_features])

	#Weights  (100 X n_hidden)
	W_hidden = tf.Variable(tf.random_uniform((n_features,n_hidden)))

	#Biases
	b_hidden = tf.Variable(tf.zeros([n_hidden]))

	#Hidden Layer Output  => (None X 20)
	h = tf.nn.tanh(tf.matmul(X,W_hidden) + b_hidden)
	
	#Output Layer Weights
	W_output = tf.transpose(W_hidden)

	#Output Layer Bias
	b_output = tf.Variable(tf.zeros([n_features]))
	
	#Final Layer Output
	prediction = tf.nn.tanh(tf.matmul(h,W_output) + b_output)

	#Actual Data
	Y = tf.placeholder(tf.float32,[None,n_features])

	#Compute the sparsity
	sparsity = np.repeat([0.05], n_hidden).astype(np.float32)
	
	#Adding L2 penalty
	regularizers = tf.nn.l2_loss(W_hidden) + tf.nn.l2_loss(W_output)

	#Calculation of Loss - Add a loss function and compute the mean of the loss along with L2 regularization
	loss =  0.5 * tf.reduce_mean(tf.pow(tf.subtract(prediction,Y),2))  + 0.01*regularizers

	#Training using Gradient Descent
	training = tf.train.AdamOptimizer(0.001).minimize(loss)

	#Initialisation Step
	init = tf.initialize_all_variables()

	with tf.Session() as sess:
		#Initialise Session Variables
		sess.run(init)

		Y_plt = []
		X_plt = []

		#Number of Rounds of Training
		n_rounds = 10000

		#Batch size -> Max Batch Size : 373
		batch_size = 200

		for i in range(n_rounds):
			#Generate an array of random numbers from pool of 0 to n_samples
			batch_indexes = np.random.randint(n_samples,size=batch_size)

			#Input Data
			input_x = input_data[batch_indexes][:]

			#Output 
			output_x = output_data[batch_indexes][:]

			#Train the model
			sess.run(training,feed_dict={X:input_x,Y:output_x})


			if i%50 == 0:
				Y_plt.append(sess.run(loss,feed_dict= {X:input_x,Y:output_x}))
				print sess.run(loss,feed_dict= {X:input_x,Y:output_x})
				
				X_plt.append(i)

		weights = sess.run(W_hidden)
		biases = sess.run(b_hidden)



	#plt.figure("Loss Plot")
	plt.plot(np.array(X_plt),np.array(Y_plt))
	plt.show()


	return weights, biases 



def main():
	#TF List and Data Matrix
	transcription_factors, data_matrix = create_data_matrix()

	#Transpose Matrix
	data_matrix = data_matrix.astype(float)
	data_matrix = data_matrix.transpose()

	#visualise(data_matrix)

	#Convert into Normal Distribution
	data_matrix = gaussian(data_matrix)
	
	#Get a latent representation of data -> Parameters : Number of Hidden Units
	weights, biases = train(data_matrix,50)

	reduced_matrix = np.matmul(data_matrix,weights) + biases
	
	visualise(data_matrix)
	visualise(reduced_matrix)












	return





main()