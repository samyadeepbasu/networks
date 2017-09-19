"""  Denoising Autoencoders to get a latent representation of the expression levels  """


import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
from matplotlib import collections as mc
import math 
import tensorflow as tf
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.manifold import Isomap, SpectralEmbedding,TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
import networkx as nx
from shapely.geometry import LineString, Point

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
def visualise(data_matrix,color):
	#Reduce Dimensions using PCA
	pca = KernelPCA(n_components=2).fit_transform(data_matrix)
	X = pca[:,0]
	Y = pca[:,1]
	color_map = [0,2,5,20,22]

	#plt.scatter(X,Y,c=color*2)
	#plt.show()

	return pca

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

	#Adding Noise to th
	input_data = input_data + 0.2 * np.random.random_sample((input_data.shape))

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
		batch_size = 300

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

#Function to construct the Minimum Spanning Tree
def get_minimum_spanning_tree(cluster_centres,labels):
	#Construct Weight Matrix
	weight_matrix = np.zeros((len(labels),len(labels)))

	for i in range(len(labels)):
		for j in range(len(labels)):
			if i==j:
				weight_matrix[i][j] = 0 
				weight_matrix[j][i] = 0

			else:
				weight_matrix[i][j] = math.sqrt((cluster_centres[i][0] - cluster_centres[j][0])**2  + (cluster_centres[i][1] - cluster_centres[j][1])**2 )
				weight_matrix[j][i] = weight_matrix[i][j]


	#Add to the NetworkX Graph
	g = nx.Graph()

	for i in range(len(labels)):
		for j in range(len(labels)):
			g.add_edge(i,j,weight=weight_matrix[i][j])

	#Minimum Spanning Tree
	T=nx.minimum_spanning_tree(g)

	return T.edges()


#Function to get the cluster centres by combining with prior biological knowledge
def cluster_centres(reduce_dim,times):
	#Get Unique Labels
	unique_labels = np.unique(times)

	cluster_centres = []

	for label in unique_labels:
		#Indexes for the label
		indexes = np.where(times == label)[0]
		
		#Get the cells
		cells = reduce_dim[indexes][:]

		#Get the cluster centre
		x = cells[:,0]
		y = cells[:,1]

		average_x = np.mean(x)
		average_y = np.mean(y)

		cluster_centres.append([average_x,average_y])


	cluster_centres = np.array(cluster_centres)
	#print cluster_centres

	""" Construction of MST without using Prior Knowledge about Cell Collection, but prior knowledge about cell collection used in clustering """
	#MST Connections with minimum weight
	connections = get_minimum_spanning_tree(cluster_centres,unique_labels)

	#Create line segments
	line_segments = []

	for edge in connections:
		line_segments.append([(cluster_centres[edge[0]][0],cluster_centres[edge[0]][1]),(cluster_centres[edge[1]][0],cluster_centres[edge[1]][1])])


	lc = mc.LineCollection(line_segments, colors='black', linewidths=2)
	

	plt.scatter(reduce_dim[:,0],reduce_dim[:,1],c=times)
	plt.scatter(cluster_centres[:,0],cluster_centres[:,1],140,c='black',marker='x')
	for i in range(len(line_segments)):
		plt.plot(np.array([line_segments[i][0][0],line_segments[i][1][0]]),np.array([line_segments[i][0][1],line_segments[i][1][1]]),c='black',linewidth=2)


	plt.show()


	return line_segments,cluster_centres,unique_labels


#Function to obtain the actual time
def actual_cell_time():
	#Open File
	f = open('data2/time.txt','r')

	times = [line.split()[2] for line in f.readlines()]

	f.close()

	return np.array(times).astype(float)


#Function to get the projections
def get_projections(lines,cluster_centre,unique_labels,times,matrix):
	""" Algorithm : 
			   For each point -> find lines originating from cluster centres, project on the line where the distance is shortest """

	projections = []

	for i in range(len(matrix)):
		#print matrix[i]
		#print times[i]
		index = np.where(unique_labels == times[i])[0]

		#Cluster Centre for the current point
		centre = (cluster_centre[index[0]][0],cluster_centre[index[0]][1])

		#Extract all the lines originating from the cluster centre
		temp_lines = []


		for line in lines:
			if centre in line:
				temp_lines.append(line)

		distances = []

		for line in temp_lines:
			#Get the line from the cluster centre
			l = LineString(line)

			#Define Point
			q = Point(matrix[i][0],matrix[i][1])

			distances.append(q.distance(l))


		#Obtain the Minimum Distance and the line corresponding to it --> Then project the point onto the line
		min_distance_index = distances.index(min(distances))

		#Line on which the projection has to take place
		projection_line = temp_lines[min_distance_index]

		p_l = LineString(projection_line)

		ns = p_l.interpolate(p_l.project(Point(matrix[i][0],matrix[i][1])))

		projections.append(np.array(ns))		


	projections = np.array(projections)

	X = projections[:,0]
	Y = projections[:,1]

	plt.scatter(X,Y,c=times, s=130,alpha=0.4)
	plt.show()
	#plt.clf()

	""" Projection of branches with the starting point mentioned """
	
	#Known Biological Knowledge
	starting_point = (cluster_centre[0][0],cluster_centre[0][1])

	temp_point = (cluster_centre[1][0],cluster_centre[1][1])

	#Get the line for the starting point
	for line in lines:
		if starting_point in line:
			first_line = line

		if temp_point in line:
			second_line = line


	#Construct Line String
	l_s = LineString(first_line)

	#Get the Line with has to be projected
	projection_line = LineString(second_line)
	
	new_projections = []
	old_projections = []
	
	i = 0
	temp = []
	for point in projections:
		temp_point = Point((point[0],point[1]))
	
		if projection_line.distance(temp_point) < 0.00000000001 :
			#Project point onto first line 
			a = np.array(l_s.interpolate(l_s.project(temp_point)))
			new_projections.append(a)		
			temp.append(times[i])


		else:
			new_projections.append(point)

		i += 1

	new_projections = np.array(new_projections)
	old_projections = np.array(old_projections)

	x = new_projections[:,0]
	y = new_projections[:,1]
	 
	#plt.cla()
	plt.scatter(x,y,c=times, s=130,alpha=0.4)
	plt.show()
	
	#plt.scatter(old_projections[:,0],old_projections[:,1],s=130,alpha=0.4)

	return


def monocle_time():
	f = open('data2/time.txt','r')

	times = [[line.split()[1],line.split()[2]] for line in f.readlines() ]

	times = np.array(times).astype(float)
	X = times[:,0]
	color = times[:,1]
	
	Y = np.repeat(2,len(X))
	plt.scatter(X,Y,c=color,s=140,alpha=0.3)
	plt.show()

	return 



def main():
	#TF List and Data Matrix
	transcription_factors, data_matrix = create_data_matrix()

	#Transpose Matrix
	data_matrix = data_matrix.astype(float)
	data_matrix = data_matrix.transpose()
	old_matrix = data_matrix.copy()

	#visualise(data_matrix)
	
	#Convert into Normal Distribution
	#data_matrix = gaussian(data_matrix)

	#Get a latent representation of data -> Parameters : Number of Hidden Units
	#weights, biases = train(data_matrix,500)

	#reduced_matrix = np.matmul(data_matrix,weights) + biases
	
	#temp_matrix = PCA(n_components=5).fit_transform(old_matrix)
	
	times = actual_cell_time()	

	#visualise(old_matrix,times)
	reduce_dim = visualise(old_matrix,times)
	#reduce_dim = visualise(reduced_matrix,times)

	#temp_deep = gaussian(reduce_dim) 

	#weights, biases = train(temp_deep,2)

	#reduced_matrix = np.matmul(temp_deep,weights) + biases


	
	lines, cluster_centre, unique_labels = cluster_centres(reduce_dim,times)
	
	#Get Projections for the points on the main line
	get_projections(lines,cluster_centre,unique_labels,times,reduce_dim)

	monocle_time()


	

	#print actual_cell_time()
	#monocle_time()

	#projection_points = get_projections(reduce_dim,lines)
	
	""" 
	index_0 = np.where(times==22)
	cells_0 = reduce_dim[index_0][:]
	k_mean = AgglomerativeClustering(n_clusters=4).fit_predict(cells_0)
	#print k_mean
	lines = cluster_centres(cells_0,k_mean)

	#plt.scatter(cells_0[:,0],cells_0[:,1],c=k_mean)
	#plt.show()
	"""


	return 

	#plt.scatter(np.array([0,1,2,3,4]),np.array([0,1,2,3,4]),c=np.array([0,4,10,40,44]))
	#plt.show()





main()