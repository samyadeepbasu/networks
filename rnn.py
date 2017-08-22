""" Reconstruction of regulatory networks from sc-RNA seq Data using Recurrent Neural Networks with Long Short Term Memory  

    1. Baseline Method using RNN without Long Short Term Memory  -- Vanilla RNN                                                     """


from __future__ import  division
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
from noise_reduction import clustered_state, matrix_pass


#Initialisations for Test
num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length


#Function to Generate Data for the Testing the Network -- Type of Data : Rolled Sequence
def generateData():
    #Generation of numbers from np.arange(2) with a size of total_series_length, probability of drawing a number is 0.5
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))

    #Roll the numbers by steps which will be the desired output
    y = np.roll(x, echo_step)

    #Initialising the first few echo steps after rolling
    y[0:echo_step] = 0
    
    #Converting the training data into batches
    x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1))

    return (x, y)


#Function to curate the data for the network
def get_expression_data():
    #Generate a time series dataset, parameter : Number of States
    data_matrix, cells = matrix_pass()

    data_matrix = data_matrix.transpose()

    """ Expression levels of the first state are given as input """

    #Input to the Recurrent Neural Network
    input_data = data_matrix[0]

    #Generate the Labels
    labels = data_matrix[1:]

    return input_data, labels


#Function to implement the Recurrent Net for Modelling the Expression
def recurrent_net():
    #Training Data
    input_matrix , labels = get_expression_data()

    #Initial State of the Network will be the expression level of the first state -- To be reshaped
    current_state = input_matrix

    #Number of Neurons in Each State --> Number of Genes in one state
    state_size = len(current_state)

    #Reshaping the current matrix for matrix multiplication
    current_state = np.reshape(current_state,(1,100))    

    #Define the TensorFlow Variables for State Transitions
    W = tf.Variable(np.random.rand(state_size,state_size),dtype=tf.float32)
    b = tf.Variable(np.zeros((1,state_size)),dtype=tf.float32)

    #Define the Tensorflow Variables for Output and Training
    W2 = tf.Variable(np.random.rand(state_size,state_size),dtype=tf.float32)
    b2 = tf.Variable(np.zeros((1,state_size)),dtype=tf.float32)

    #Define the placeholder for state - Dimension : (1 * 100)
    initial_state = tf.placeholder(tf.float32,[1,state_size])

    #Start the Tensorflow Session
    with tf.Session() as sess:
        #Initialize all tensorflow variables
        sess.run(tf.initialize_all_variables())

        for i in range(num_epochs):
            #In each epoch train the whole network
            print len(sess.run(W2))

            print initial_state.get_shape()

            print current_state

            break

        







recurrent_net()


    


#Input Data & Labels
input_data,labels = get_expression_data()





batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

init_state = tf.placeholder(tf.float32, [batch_size, state_size])

#Tensor flow variables
W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)
b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)


#Build Graph that resembles the RNN architecture -- Unpacks the columns of the matrix into lists
input_series = tf.unpack(batchX_placeholder,axis=1)
label_series = tf.unpack(batchY_placeholder,axis=1)


#Defining and initialising the states
current_state = init_state
states_series = []

for current_input in input_series:
    #Reshape the current matrix
    current_input = tf.reshape(current_input,[batch_size,1])

    #Current Input and State Matrix concatenated -- Assuming both have the same weights
    concatenated_input = tf.concat(1,[current_input,current_state])

    #Calculating the next state -- The bias is broadcasted across all the samples
    next_state = tf.tanh(tf.matmul(concatenated_input,W) + b) 

    #Append to the states series
    states_series.append(next_state)
    
    #Initialise the next state instance
    current_state = next_state


#Output Series for Loss Function Calculation and Training
output_series = [tf.matmul(state,W2) + b2 for state in states_series]

#Pass the Output through a Softmax layer
predictions = [tf.nn.softmax(output) for output in output_series]

#Calculation of the losses for each batch
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits,labels) for logits,labels in zip(output_series,label_series)]

#Compute the total loss
total_loss = tf.reduce_mean(losses)

#Train the Model using the Computed Loss
train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)


"""
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    loss_list = []

    #print sess.run(W)
    #print sess.run(b)

    for epochs in range(num_epochs):
        #Generation of training set
        x,y = generateData()

        _current_state = np.zeros((batch_size,state_size))

        #print __current__state

        print("New data, epoch", epochs)

        for i in range(num_batches):
            start_pos = i*truncated_backprop_length
            end_pos = start_pos + truncated_backprop_length

            batchX = x[:,start_pos:end_pos]
            batchY = y[:,start_pos:end_pos]

            _total_loss, _train_step, _current_state, _output_series = sess.run(
                [total_loss, train_step, current_state, output_series],
                feed_dict={
                    batchX_placeholder:batchX,
                    batchY_placeholder:batchY,
                    init_state:_current_state
                })

            loss_list.append(_total_loss)


            if i%100 == 0 :
                print "Loss"
                print("Step",i, "Loss", _total_loss)




    #print loss_list

    plt.plot(loss_list)
    plt.show()
        

        

"""







#print x[0]

