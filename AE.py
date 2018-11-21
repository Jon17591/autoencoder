import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
import time

class AutoEncoder:
    def __init__(self,learning_rate,struct,name='AE'):
        self.learning_rate=learning_rate
        self.struct=struct
        with tf.variable_scope(name):
            self.inputs_=tf.placeholder(tf.float32,[None,self.struct[0]],name='inputs')
            self.weights={
                'encoder_h1':tf.Variable(tf.random_normal([self.struct[0],self.struct[1]])),
                'decoder_h1':tf.Variable(tf.random_normal([self.struct[1],self.struct[2]]))
                }
            self.biases={
                'encoder_b1':tf.Variable(tf.random_normal([self.struct[1]])),
                'decoder_b1':tf.Variable(tf.random_normal([self.struct[2]])),
                }
            self.encoder_in=tf.nn.sigmoid(tf.add(tf.matmul(self.inputs_,self.weights['encoder_h1']),self.biases['encoder_b1']))
            self.decoder_out=tf.nn.sigmoid(tf.add(tf.matmul(self.encoder_in,self.weights['decoder_h1']),self.biases['decoder_b1']))
            self.y_pred=self.decoder_out
            self.y_true=self.inputs_
            self.cost=tf.reduce_mean(tf.pow(self.y_true-self.y_pred,2))
            self.opt=tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)

def Demonstrate():
    ae=AutoEncoder(0.01,[4,2,4])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            _,c=sess.run([ae.opt,ae.cost],feed_dict={ae.inputs_:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]})
            print('Cost: %s - %s'%(i,c))
        print('finished')
        encode_decode=sess.run(ae.decoder_out,feed_dict={ae.inputs_:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]})
    return encode_decode
	
if __name__ == '__main__':	
    time_start=time.time()
    print(Demonstrate())
    print('Time elapsed: %s'%(time.time()-time_start))


