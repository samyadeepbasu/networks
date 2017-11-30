""" Algorithm to reconstruct regulatory networks by learning the latent features associated with a node using Deep Learning Models"""

#############################################################################################################

#Libraries
import math
import numpy as np
import random
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor, GradientBoostingRegressor
from sklearn import metrics
import networkx as nx
from numpy import inf
from scipy.stats import pearsonr, spearmanr
from scipy import spatial
from sklearn.metrics import mutual_info_score
from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean
from scipy.spatial.distance import euclidean
import tensorflow as tf
from fastdtw import fastdtw
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from sklearn import svm
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge



###############################################################################################################
###############################################################################################################


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


#Function to find the unique transcriptional factors 
def find_unique_tfs(interactions):
	temp = []

	for interaction in interactions:
		temp.append(interaction[0])
		temp.append(interaction[1])

	temp = list(set(temp))

	return temp

#Function to get the ground truth for the dataset
def ground_truth():
	#Open and Initialise the File
	g_file = open('ground_truth/stamlab_for_data2.txt','r')

	#Conversion of the interactions in appropriate format  -- (Regulator --->  Target)
	interactions = [ (int(line.split()[3]),int(line.split()[2])) for line in g_file.readlines()]

	for item in interactions: #Remove Self-Loops
		if item[0] == item[1]:
			interactions.remove(item)
	
	return interactions

#Function to generate the negative samples
def get_negative_interactions(positive_interactions):
	#Find the unique transcription factors
	unique_tfs = find_unique_tfs(positive_interactions)

	total_sample = []

	for regulator in unique_tfs:
		for target in unique_tfs:
			if target != regulator:
				total_sample.append((regulator,target))


	negative_samples = list(set(total_sample) - set(positive_interactions))

	return total_sample, negative_samples


#Function to generate the similarity measurement
def similarity(vec1,vec2):
	return rbf_kernel(vec1.reshape(1,-1),vec2.reshape(1,-1))
	
	#return mutual_info_score(vec1,vec2)


#Convert Data into Gaussian Distribution before passing to AutoEncoder
def gaussian(data_matrix):
	for i in range(len(data_matrix)):
		data_matrix[i] = (data_matrix[i] - np.mean(data_matrix)) / np.std(data_matrix)
		#data_matrix[i] = (data_matrix[i] - max(data_matrix[i])) / (max(data_matrix[i]) - min(data_matrix[i]))

	return data_matrix


#Function to split into training and testing sample
def split(total_samples):
	
	number_edges = len(total_samples)
	
	number_testing = int(0.2*number_edges)

	#Do a random shuffle
	random.shuffle(total_samples)

	chunks = []
	
	for i in range(0,len(total_samples),number_testing):
		chunks.append(total_samples[i:i+number_testing])


	return chunks	


#Function to train an autoencoder and generate a latent representation of the cells
def train(data_matrix,k):
	#Input without noise
	input_data = data_matrix

	#Output data
	output_data = input_data

	#Adding Noise to th
	input_data = input_data + 0.1* np.random.random_sample((input_data.shape))

	#Keep Probability for Dropouts
	keep_prob = tf.placeholder(tf.float32)

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

	#Adding Dropout
	drop_out = tf.nn.dropout(h, keep_prob)
	
	#Output Layer Weights
	W_output = tf.transpose(W_hidden)

	#Output Layer Bias
	b_output = tf.Variable(tf.zeros([n_features]))
	
	#Final Layer Output
	prediction = tf.nn.tanh(tf.matmul(drop_out,W_output) + b_output)

	#Actual Data
	Y = tf.placeholder(tf.float32,[None,n_features])

	#Compute the sparsity
	sparsity = np.repeat([0.05], n_hidden).astype(np.float32)
	
	#Adding L2 penalty
	regularizers = tf.nn.l2_loss(W_hidden) + tf.nn.l2_loss(W_output) + tf.nn.l2_loss(b_hidden) + tf.nn.l2_loss(b_output)

	#Calculation of Loss - Add a loss function and compute the mean of the loss along with L2 regularization
	loss =  0.5 * tf.reduce_mean(tf.pow(tf.subtract(prediction,Y),2))  + 0.01 * regularizers

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
		n_rounds = 5000

		#Batch size -> Max Batch Size : 373
		batch_size = 30

		for i in range(n_rounds):
			#Generate an array of random numbers from pool of 0 to n_samples
			batch_indexes = np.random.randint(n_samples,size=batch_size)

			#Input Data
			input_x = input_data[batch_indexes][:]

			#Output 
			output_x = output_data[batch_indexes][:]

			#Train the model
			sess.run(training,feed_dict={X:input_x,Y:output_x,keep_prob:0.5})


			if i%50 == 0:
				Y_plt.append(sess.run(loss,feed_dict= {X:input_x,Y:output_x,keep_prob:0.5}))
				print sess.run(loss,feed_dict= {X:input_x,Y:output_x,keep_prob:0.5})
				
				X_plt.append(i)

		weights = sess.run(W_hidden)
		biases = sess.run(b_hidden)



	#plt.figure("Loss Plot")
	plt.plot(np.array(X_plt),np.array(Y_plt))
	plt.show()


	return weights, biases 


def get_scores(total_samples,test_graph,data_matrix,similarity_matrix,unique_tfs):
	reduced_matrix = similarity_matrix

	#weights, biases = train(similarity_matrix,100)

	#reduced_matrix = np.matmul(similarity_matrix,weights) + biases


	edge_scores = []

	#Create a model
	for edge in test_graph:
		regulator = edge[0]

		targets = []

		for link in total_samples:
			if link[0] == regulator:
				targets.append((reduced_matrix[unique_tfs.index(link[1])],link[2]))

		X = np.array([target[0] for target in targets])

		Y = np.array([target[1] for target in targets])

		#Calculate similarity
		current_target_expression = reduced_matrix[unique_tfs.index(edge[1])]

		#clf = svm.SVR(kernel='rbf')
		#clf = LinearRegression()
		#clf = GradientBoostingRegressor(n_estimators=30)
		clf = Lasso(alpha=10)
		clf.fit(X,Y)
		predict_score = clf.predict(current_target_expression.reshape(1,-1))

		""" Local Model """

		n_h = []

		#Local model for classification
		for link in total_samples:
			if link[0] == regulator:
				n_h.append((data_matrix[link[1]],link[2]))


		X = np.array([target[0] for target in n_h])

		Y = np.array([target[1] for target in n_h])

		temp = np.where(Y==1)
		

		#clf = svm.SVR(C=100,kernel='rbf')
		#clf = LinearRegression()
		#clf = GradientBoostingRegressor(n_estimators=30)
		clf = Lasso(alpha=10)
		clf.fit(X,Y)

		predict_score_2 = clf.predict(data_matrix[edge[1]].reshape(1,-1))

		net_score = float(predict_score[0] + predict_score_2[0]) / 2

		#edge_scores.append(predict_score[0])

		edge_scores.append(net_score)



	edge_scores = np.array(edge_scores)
	print edge_scores

	true_labels = [edge[2] for edge in test_graph]

	score = metrics.roc_auc_score(true_labels,edge_scores)
	
	print "#"
	print score

	fpr, tpr, thresholds = metrics.roc_curve(true_labels,edge_scores,drop_intermediate=False)

	fig, ax = plt.subplots()
	ax.plot(np.array(fpr),np.array(tpr), c='black')
	line = mlines.Line2D([0, 1], [0, 1], color='red')
	transform = ax.transAxes
	line.set_transform(transform)
	ax.add_line(line)
	plt.show()

	return score


	


#Function to create a graph
def create_graph(total_samples,data_matrix):
	#Regulators
	similarity_matrix = []

	#Create a training graph for the positive interactions
	unique_tfs = find_unique_tfs(total_samples)

	print len(total_samples)

	for gene in unique_tfs:
		#Extract it's neighbours
		vector = np.zeros((len(unique_tfs)))

		for neighbour in total_samples:
			if neighbour[0] == gene and neighbour[2]==1:
				#Get the target 
				target_index = unique_tfs.index(neighbour[1])

				#Get the similarity
				vector[target_index] = 1 #similarity(data_matrix[gene],data_matrix[neighbour[1]])





		similarity_matrix.append(vector)	


	similarity_matrix = np.array(similarity_matrix)
	print similarity_matrix

	#Split into Training and Testing Data
	splitted_sample = split(total_samples)
	splitted_sample = splitted_sample[:5]


	AUC = []

	for i in range(0,len(splitted_sample)):
		testing_set = splitted_sample[i]
		training_set_index = range(0,len(splitted_sample))
		training_set_index.remove(i)
		training_set = []
		for index in training_set_index:
			training_set += splitted_sample[index]

		AUC.append(get_scores(training_set,testing_set,data_matrix,similarity_matrix,unique_tfs))



	return AUC


def main():
	transcription_factors, data_matrix = create_data_matrix()

	data_matrix = data_matrix.astype(float)

	#Positive Interactions
	positive_interactions = ground_truth()

	#Ground Truth : Negative Interactions : (Regulator, Target)
	total_samples, negative_interactions = get_negative_interactions(positive_interactions)
	
	positive_samples = [link + (1,) for link in positive_interactions]
	negative_samples = [link + (0,) for link in negative_interactions]
	total_samples = positive_samples + negative_samples

	random.shuffle(total_samples)

	#Split into Training and Testing Data
	splitted_sample = split(total_samples)
	splitted_sample = splitted_sample[:5]

	auc = create_graph(total_samples,data_matrix)

	print np.mean(np.array(auc))

	

	return



main()