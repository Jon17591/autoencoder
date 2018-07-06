"""
Created on Tue Jul  3 18:00:41 2018

@author: jt17591

messing about with neural networks or atleast trying to
"""

import numpy as np 
import matplotlib.pyplot as plot
from time import time 

class Network:
    
    def __init__(self,input_nodes,hidden_nodes,output_nodes):
        """
        Define the architecture and create the relevant input and output 
        matrices. These matrices represent the edges between the nodes.
        """
        self.input_nodes=input_nodes
        self.hidden_nodes=hidden_nodes
        self.output_nodes=output_nodes
        self.weight_mat_in=self.Initialise_Weight_Mat(self.input_nodes+1,self.hidden_nodes)
        self.weight_mat_out=self.Initialise_Weight_Mat(self.hidden_nodes+1,self.output_nodes)
        
    def Initialise_Weight_Mat(self,rows,columns):
        """
        Randomly initialise a matrix of the specified dimensions
        """
        return np.random.rand(rows,columns)
        
    def Layer_In(self,data):
        """
        Compute the values of the hidden layer. Multiple input by the matrix 
        representing the weights between input and hidden layer. Then use 
        sigmoid as the activation function for the neurons. Note a bias is 
        added to the data input so that the neurons can have a bias. 
        """
        add_bias=np.append(data,1) #add bias for the nodes in the hidden layer. 
        result=np.matmul(add_bias,self.weight_mat_in)
        self.hidden_return=self.Sigmoid_Fn(result)
        return self.hidden_return
    
    def Layer_Out(self,hidden_in):
        """
        Compute the values of the output layer. Multiple hidden layer by the 
        matrix representing the edges between hidden and output. As before a
        bias is added to the neurons in the form of a extra neuron at the 
        hidden layer which is not connected to the input layer. 
        """
        add_bias=np.append(hidden_in,1)
        result=np.matmul(add_bias,self.weight_mat_out)
        self.out_return=self.Sigmoid_Fn(result)
        return self.out_return
    
    def Sigmoid_Fn(self,val):
        """
        Sigmoid activation function. So that the output is a value between 0 
        and 1. This is also differentiatable which allows for gradient descent. 
        """
        return np.divide(1.0,(1.0+np.exp(-val)))
    
    def Minimise_Error(self,data,expected,actual):
        """
        Initialise backpropagation algorithm on the weights of the network. 
        First loop is for weights between hidden and output. 
        Second loop is for weights between input and hidden. 
        """
        for i in range(self.output_nodes):
            for j in range(len(self.weight_mat_out[:,i])):#rows
                change_out=np.append(self.hidden_return,1)[j]*actual[i]*(1-actual[i])*(expected[i]-actual[i])
                self.weight_mat_out[j,i]=self.weight_mat_out[j,i]+(0.5*change_out)
        for i in range(self.input_nodes+1):
            for j in range(self.hidden_nodes):
                change_in=np.append(data,1)[i]*self.hidden_return[j]*(1-self.hidden_return[j])*np.matmul(self.weight_mat_out[j,:],(expected-actual))
                self.weight_mat_in[i,j]=self.weight_mat_in[i,j]+(0.5*change_in)

def Loss_Function(expected,actual):
    """
    Squared sum error, which allows progress of the convergence to be monitored.
    """
    return np.sum(np.square(expected-actual))

a=Network(4,2,4) #Initialise network arguments are input dimension, hidden dimension and output dimension. 
data=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) #My codewords which the autoencoder is trying to learn a mapping for
loss=[] #record loss values
start_time=time() #allows for measuring of elapsed time. 
for i in range(5000): 
    loss_temp=[] #loss log for this iteration through data
    for i in range(len(data)):
        actual=a.Layer_Out(a.Layer_In(data[i])) #pass data through neural network and calculate the value it thinks it should be. 
        loss_temp.append(Loss_Function(data[i],actual)) #record loss
        a.Minimise_Error(data[i],data[i],actual) #minimise loss
    loss.append(np.average(loss_temp)) #record average loss over codewords
print("The elapsed time is %s"%(time()-start_time)) 

plot.plot(loss) #plot the improvement of loss over each backpropagation iteration. 
plot.title("Mean Squared Error of autoencoder")
plot.xlabel("Iterations")
plot.ylabel("MSE")
"""
This is a test run, just to confirm that the autoencoder does infact produce the same inputs that were passed in. 
"""
for i in range(len(data)):
    actual=a.Layer_Out(a.Layer_In(data[i]))
    print(np.round(actual,decimals=1))
    print(Loss_Function(data[i],actual))
    



