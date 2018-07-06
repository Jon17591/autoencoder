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
        self.input_nodes=input_nodes
        self.hidden_nodes=hidden_nodes
        self.output_nodes=output_nodes
        self.weight_mat_in=self.Initialise_Weight_Mat(self.input_nodes+1,self.hidden_nodes)
        self.weight_mat_out=self.Initialise_Weight_Mat(self.hidden_nodes+1,self.output_nodes)
        
    def Initialise_Weight_Mat(self,rows,columns):
        return np.random.rand(rows,columns)
        
    def Layer_In(self,data):
        add_bias=np.append(data,1) #add bias
        result=np.matmul(add_bias,self.weight_mat_in)
        self.hidden_return=self.Sigmoid_Fn(result)
        return self.hidden_return
    
    def Layer_Out(self,hidden_in):
        add_bias=np.append(hidden_in,1)
        result=np.matmul(add_bias,self.weight_mat_out)
        self.out_return=self.Sigmoid_Fn(result)
        return self.out_return
    
    def Sigmoid_Fn(self,val):
        return np.divide(1.0,(1.0+np.exp(-val)))
    
    def Minimise_Error(self,data,expected,actual):
        for i in range(self.output_nodes):#columns output
            for j in range(len(self.weight_mat_out[:,i])):#rows
                change_out=np.append(self.hidden_return,1)[j]*actual[i]*(1-actual[i])*(expected[i]-actual[i])
                self.weight_mat_out[j,i]=self.weight_mat_out[j,i]+(0.75*change_out)
        for i in range(self.input_nodes+1):
            for j in range(self.hidden_nodes):
                change_in=np.append(data,1)[i]*self.hidden_return[j]*(1-self.hidden_return[j])*np.matmul(self.weight_mat_out[j,:],(expected-actual))
                self.weight_mat_in[i,j]=self.weight_mat_in[i,j]+(0.75*change_in)
                
                
                

def Loss_Function(expected,actual):
    return np.sum(np.square(expected-actual))

a=Network(4,2,4)
data=np.array([[0,0],[0,1],[1,0],[1,1]])
outputs=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
loss=[]
start_time=time()
for i in range(5000):
    for i in range(len(data)):
        actual=a.Layer_Out(a.Layer_In(outputs[i]))
        loss.append(Loss_Function(outputs[i],actual))
        a.Minimise_Error(outputs[i],outputs[i],actual)
        
print("The elapsed time is %s"%(time()-start_time))
plot.plot(loss)

for i in range(len(data)):
    actual=a.Layer_Out(a.Layer_In(outputs[i]))
    print(np.round(actual,decimals=1))
    print(Loss_Function(outputs[i],actual))
    



