""" Baseline Method for using the Prior Graph to predict edges, when combined with gene expression  """

#############################################################################################################
#############################################################################################################

""" Assumptions : If multiple genes are regulated by a regulator, the targets have some similarity amongst themselves  
		Base Algorithm : 
			1. Obtain a reduced form of the prior Graph, by deleting a certain percentage of edges -> Training Data
			2. Treat the deleted edges as testing set 
			3. For each edge in the testing set, score the edge according to the scheme 
			  Scoring scheme : 
				-> For the edge in the test set, find the regulator and it's corresponding targets 
				-> Find the similarity of the target in the test edge with that of the remaining targets in the training targets
				-> Churn out the scores corresponding to the top k similar neighbours

			Ideally : 
				-> A potential edge will have a high score and a negative edge should have a low score 

			For each testing fold generate an AUPR curve and average it out overall the folds to generate the final curve                                                 
																															  """
#############################################################################################################
#############################################################################################################


#Libraries
import math
import numpy as np
import random
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.metrics.pairwise import rbf_kernel,polynomial_kernel,sigmoid_kernel,laplacian_kernel,chi2_kernel
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
from fastdtw import fastdtw
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from sklearn import svm
import tensorflow as tf 
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge,ElasticNet
import networkx as nx



###############################################################################################################
###############################################################################################################

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

	return tf_list, data_matrix_array


#Function to order cells by pseudo time
def pseudo_time(data_matrix):
	#Open the file corresponding to Pseudo Time Measurement
	time_file = open('data3/time.txt','r')
	
	#Extraction of Pseudo Time from the List
	ordered_cells = [line.split()[1] for line in time_file.readlines()]

	#ordered_cells = order_time() #Using Imputation Algorithm and TSCAN

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


#Function to get a list of times
def time():
	#Time Measurements
	time_series = open('data3/time.txt','r')

	times = [line.split()[1] for line in time_series.readlines()]

	#times = order_time()  #Using Imputation Algorithm and TSCAN

	sorted_time = np.sort(np.array(times).astype(float))

	#Normalise
	#normalised_time = (sorted_time - np.min(sorted_time)) / (np.max(sorted_time) - np.min(sorted_time))
	normalised_time = sorted_time / max(sorted_time)
	#print normalised_time


	return normalised_time


#Function to get the ground truth for the dataset
def ground_truth():
	#Open and Initialise the File
	g_file = open('ground_truth/stamlab_for_data3.txt','r')

	#Conversion of the interactions in appropriate format  -- (Regulator --->  Target)
	interactions = [ (int(line.split()[3]),int(line.split()[2])) for line in g_file.readlines()]

	for item in interactions: #Remove Self-Loops
		if item[0] == item[1]:
			interactions.remove(item)
	
	return interactions

#Function to extract the targets
def get_targets(regulator,interactions):
	targets = []

	for edge in interactions:
		if edge[0] == regulator:
			targets.append(edge[1])

	return targets

#Function to find the unique transcriptional factors 
def find_unique_tfs(interactions):
	temp = []

	for interaction in interactions:
		temp.append(interaction[0])
		temp.append(interaction[1])

	temp = list(set(temp))

	return temp



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



#Function to get the minimum of three numbers
def find_minimum(a,b,c):
	minimum_list = []
	minimum_list.append(a)
	minimum_list.append(b)
	minimum_list.append(c)

	min_list = sorted(minimum_list)

	return min_list[0]


#Function to find the distance
def distance(current_target_expression,data_matrix,targets,i):
	#return spearmanr(current_target_expression,data_matrix[targets[i]])[0]
	#return spatial.distance.correlation(current_target_expression,data_matrix[targets[i]])
	#return 1/mutual_info_score(current_target_expression,data_matrix[targets[i]])
	#return rbf_kernel(current_target_expression.reshape(1,-1),data_matrix[targets[i]].reshape(1,-1))
	#return DTW_distance(current_target_expression,data_matrix[targets[i]])

	#D = SquaredEuclidean(current_target_expression.reshape(1,-1),data_matrix[targets[i]].reshape(1,-1))
	#sdtw = SoftDTW(D, gamma=1.0)
	distance, path = fastdtw(current_target_expression.reshape(1,-1),data_matrix[targets[i]].reshape(1,-1), dist=euclidean)

	#return 1/sdtw.compute()

	return 1/float(distance)



#Function to train an autoencoder and generate a latent representation of the cells
def train(data_matrix,k):
	#Input without noise
	input_data = data_matrix

	#Output data
	output_data = input_data

	#Adding Noise to th
	input_data = input_data + 0.3* np.random.random_sample((input_data.shape))

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
	regularizers = tf.nn.l2_loss(W_hidden) + tf.nn.l2_loss(W_output)# + tf.nn.l2_loss(b_hidden) + tf.nn.l2_loss(b_output)

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
		batch_size = 80

		for i in range(n_rounds):
			#Generate an array of random numbers from pool of 0 to n_samples
			batch_indexes = np.random.randint(n_samples,size=batch_size)

			#Input Data
			input_x = input_data[batch_indexes][:]

			#Output 
			output_x = output_data[batch_indexes][:]

			#Train the model
			sess.run(training,feed_dict={X:input_x,Y:output_x,keep_prob:0.3})


			if i%50 == 0:
				Y_plt.append(sess.run(loss,feed_dict= {X:input_x,Y:output_x,keep_prob:0.3}))
				print sess.run(loss,feed_dict= {X:input_x,Y:output_x,keep_prob:0.3})
				
				X_plt.append(i)

		weights = sess.run(W_hidden)
		biases = sess.run(b_hidden)



	#plt.figure("Loss Plot")
	plt.plot(np.array(X_plt),np.array(Y_plt))
	plt.show()


	return weights, biases 



#Function to create classification model for each local region
def create_classification_model(training,testing,data_matrix):
	""" Algorithm : For each edge in testing set
					  -> Go to the local neighbourhood 
					  -> Build a local classification model 
					  -> Get the score for the edge                 """
	#True Labels	
	true_labels = [edge[2] for edge in testing]


	
	#Scores based on local classification model
	edge_scores = []

	for edge in testing:
		#Get the regulator
		regulator = edge[0]

		targets = []

		for link in training:
			if link[0] == regulator:
				targets.append((data_matrix[link[1]],link[2]))


		X = np.array([target[0] for target in targets])

		Y = np.array([target[1] for target in targets])

		temp = np.where(Y==1)
		

		
		clf = svm.SVR(C=1, gamma=0.001,kernel='rbf')
		#clf = LinearRegression()
		#clf = GradientBoostingRegressor(n_estimators=30)
		#clf = Lasso(alpha=10)
		clf.fit(X,Y)

		predict_score = clf.predict(data_matrix[edge[1]].reshape(1,-1))
		
		edge_scores.append(predict_score[0])


	edge_scores = np.array(edge_scores)

	score = metrics.roc_auc_score(true_labels,edge_scores)

	print score

	fpr, tpr, thresholds = metrics.roc_curve(true_labels,edge_scores,drop_intermediate=False)

	fig, ax = plt.subplots()
	ax.plot(np.array(fpr),np.array(tpr), c='black')
	line = mlines.Line2D([0, 1], [0, 1], color='red')
	transform = ax.transAxes
	line.set_transform(transform)
	ax.set_xlabel('False Positive Rate')
	ax.set_ylabel('Recall')
	ax.add_line(line)
	plt.show()



	return score


#Function to create and initialise the model 
def create_model(training,testing,data_matrix,k):
	
	#Extract only the positive edges from the training 
	training = [edge for edge in training if edge[2]==1]
	
	edge_scores = []
	check = 0
	#Score each edge in the testing data using the training set Graph
	for edge in testing:
		#Get the score for the edge
		regulator = edge[0]
		current_target = edge[1]

		targets = []
		#Get the corresponding genes in the local neighbourhood of edge[1] which is the target
		for link in training:
			if link[0] == regulator:
				targets.append(link[1])

		

		#Score the edge with respect to the present training graph
		current_target_expression = data_matrix[current_target]

		scores = []
		
		for i in range(len(targets)):
			scores.append(distance(current_target_expression,data_matrix,targets,i))
			#scores.append(1)

		if len(scores)<k:
			if len(scores) == 0:
				edge_scores.append(0)
			else:
				edge_scores.append(float(sum(scores))/len(scores))
			#edge_scores.append(1)

		else:
			score_index_sort = np.array(list(reversed(np.argsort(scores).tolist())))
			top_k_scores = scores[:k]
			edge_scores.append(float(sum(top_k_scores))/k)


	#Once each edge got a score - Move the threshold and generate a AUPR and AUC
	auc, aupr,const_aupr = generate_graph(edge_scores,testing)

	print auc

		
	return auc



#Function to put the different states into bins
def binning(state_matrix, times, k): #Time Passed is sorted

	#temp_state_matrix = state_matrix.copy()

	#Cluster the time points
	cluster = AgglomerativeClustering(n_clusters=k)
	
	#Convert into matrix for K-means
	time_matrix = np.array([[time] for time in times])
	
	#Labels for the clusters
	labels = cluster.fit_predict(time_matrix)
	
	#Unique Labels
	ranges = []
	for i in range(0,len(labels)-1):
		if labels[i+1] != labels[i]:
			ranges.append(i)

	start = 0
	end = len(labels)

	#Create Bins for Clustering the states
	bins = []

	for i in range(len(ranges)):
		bins.append((start,ranges[i]+1))
		start = ranges[i]+1
	
	#Append the last one
	bins.append((start,end))

	""" Averaging out the states """
	
	new_state_matrix = []
	total = 0
	for bin in bins:
		temp_matrix = state_matrix[bin[0]:bin[1]]
		#Average out the expression levels for each gene
		temp = []

		
		for i in range(0,len(temp_matrix[0])):
			temp.append(np.mean(temp_matrix[:,i]))


		new_state_matrix.append(temp)


	
	new_state_matrix = np.array(new_state_matrix)

	""" Create a new time matrix """

	new_time_matrix = []

	for bin in bins:
		new_time_matrix.append(times[bin[0]])


	return new_state_matrix, new_time_matrix




""" Evaluation Scheme for the model """
#Function to generate AUC and AUPR curves
def generate_graph(edge_scores,testing):
	#Minimum score 
	min_score = min(edge_scores)

	#Maximum Score
	max_score = max(edge_scores)
	
	current_network = [(edge[0],edge[1]) for edge in testing]

	positive_edges = [(edge[0],edge[1]) for edge in testing if edge[2]==1]

	const_aupr = float(len(positive_edges)) / float(len(testing))
		
	#Increment
	increment = (max_score - min_score) / len(testing)
	
	precision = []
	recall = []
	fprs = []

	
	start = min_score

	while start <= max_score:
		#Extract the edge index above a certain threshold
		indexes = [i for i in range(len(edge_scores)) if edge_scores[i]>=start]
		
		""" Edges above threshold are the correct ones """
		#Extract the edges 
		extracted_edges = [current_network[position]+(1,) for position in indexes]

		""" All the extracted edges are true in nature or they hold the value of 1 """

		#Precision - > How many results / edges are classified as correct
		common = 0 

		for edge in extracted_edges:
			for link in positive_edges:
				if edge[0] == link[0] and edge[1] == link[1]:
					common += 1


		precision.append(float(common) / float(len(extracted_edges)))
		recall.append(float(common)/ float(len(positive_edges)))
		fprs.append(float(len(extracted_edges) - common) / float(len(testing) - len(positive_edges)))

		start += increment


	AUPR_curve = metrics.auc(np.array(recall),np.array(precision))
	AUC_curve = metrics.auc(np.array(fprs),np.array(recall))

	#plt.plot(np.array(fprs),np.array(recall))
	#plt.show()


	#Plots
	"""
	fig, ax = plt.subplots()
	ax.plot(np.array(fprs),np.array(recall), c='black')
	line = mlines.Line2D([0, 1], [0, 1], color='red')
	transform = ax.transAxes
	line.set_transform(transform)
	ax.add_line(line)
	plt.show() """

	

	return AUC_curve,AUPR_curve,const_aupr


def main():
	#Transcriptional Factors along with Data Matrix
	tf, data_matrix = create_data_matrix()

	data_matrix = data_matrix.astype(float)



	x = len(data_matrix)
	y = len(data_matrix[0])

	#data_matrix = np.random.poisson(x,y)

	#data_matrix = pseudo_time(data_matrix)

	time_series = time()

	state_range = range(300,700)

	main_matrix = data_matrix.copy()

	#weights,biases = train(main_matrix,1000)

	#new_matrix = np.matmul(main_matrix,weights) + biases 

	#main_matrix = new_matrix


	
	AUC_total = []

	for k in state_range:

		#data_matrix, time_ordered = binning(main_matrix.transpose(),time_series,80)

		#data_matrix = data_matrix.transpose()	

		#Ground Truth : Positive Interactions : (Regulator, Target)
		positive_interactions = ground_truth()
		
		#Ground Truth : Negative Interactions : (Regulator, Target)
		total_samples, negative_interactions = get_negative_interactions(positive_interactions)	
		
		#Randomly shuffle the lists before splitting into training and testing set
		random.shuffle(total_samples)
		random.shuffle(positive_interactions)
		random.shuffle(negative_interactions)

		#print positive_interactions

		#g = nx.DiGraph()

		#g.add_edges_from(positive_interactions)

		#edge_centrality = nx.edge_betweenness_centrality(g)

		#print edge_centrality

		positive_samples = [link + (1,) for link in positive_interactions]
		negative_samples = [link + (0,) for link in negative_interactions]

		total_samples = positive_samples + negative_samples

		random.shuffle(total_samples)


		#Split into Training and Testing Data
		splitted_sample = split(total_samples)
		splitted_sample = splitted_sample[:5]
		
		AUC = []
		AUPR = []
		const = []

		#for k in range(2,12):
		for i in range(0,len(splitted_sample)):
			testing_set = splitted_sample[i]
			training_set_index = range(0,len(splitted_sample))
			training_set_index.remove(i)

			training_set = []

			for index in training_set_index:
				training_set += splitted_sample[index]



			#auc, aupr,const_aupr = create_model(training_set,testing_set,main_matrix,10)
			auc = create_classification_model(training_set,testing_set,main_matrix)

			AUC.append(auc)
			#AUPR.append(aupr)
			
			



		print "Mean Score : "
		print np.mean(np.array(AUC))
		#print ""
		#AUC_total.append(np.mean(np.array(AUC)))
		break



	#X = range(0,len(AUC_total))

	#plt.plot(X,AUC_total)
	#plt.show()
	
		





main()