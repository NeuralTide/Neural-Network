# Place your EWU ID and Name here. | ID: 00953041 - Andrew Tucker

### Delete every `pass` statement below and add in your own code. 
# Implementation of the forwardfeed neural network using stachastic gradient descent via backpropagation
# Support parallel/batch mode: process every (mini)batch as a whole in one forward-feed/backtracking round trip. 

import numpy as np
import random
import math
import math_util as mu
from nn_layer import NeuralLayer
import os

class NeuralNetwork:
    
    def __init__(self):
        self.layers = []     # the list of L+1 layers, including the input layer. 
        self.L = -1          # Number of layers, excluding the input layer. 
                             # Initting it as -1 is to exclude the input layer in L. 
        self.error_list = []                     
    
    
    def add_layer(self, d = 1, act = 'tanh'):
        ''' The newly added layer is always added AFTER all existing layers.
            The firstly added layer is the input layer.
            The most recently added layer is the output layer. 
            
            d: the number of nodes, excluding the bias node, which will always be added by the program. 
            act: the choice of activation function. The input layer will never use an activation function even if it is given. 
            
            So far, the set of supported activation functions are (new functions can be easily added into `math_util.py`): 
            - 'tanh': the tanh function
            - 'logis': the logistic function
            - 'iden': the identity function
            - 'relu': the ReLU function
        '''
      
        
        self.layers = self.layers + [NeuralLayer(d, act)]
        self.layers[-1].W = []
        self.layers[-1].S = []
        self.layers[-1].X = []
        self.layers[-1].Delta = []
        self.L = self.L + 1
        
        
    

    def _init_weights(self):
        ''' Initialize every layer's edge weights with random numbers from [-1/sqrt(d),1/sqrt(d)], 
            where d is the number of nonbias node of the layer
        '''
        for l in range(1, self.L + 1):
            self.layers[l].W = np.random.randn(self.layers[l - 1].d + 1, self.layers[l].d) * np.sqrt(1 / self.layers[l - 1].d)
                
                
    
    
        
    def fit(self, X, Y, eta = 0.01, iterations = 1000, SGD = True, mini_batch_size = 10):
        ''' Find the fitting weight matrices for every hidden layer and the output layer. Save them in the layers.
          
            X: n x d matrix of samples, where d >= 1 is the number of features in each training sample
            Y: n x k vector of lables, where k >= 1 is the number of classes in the multi-class classification
            eta: the learning rate used in gradient descent
            iterations: the maximum iterations used in gradient descent
            SGD: True - use SGD; False: use batch GD
            mini_batch_size: the size of each mini batch size, if SGD is True.  
        '''
        self._init_weights()  # initialize the edge weights matrices with random numbers.
       
        x_bias_o = np.insert(X, 0, 1, axis=1)
        
        indices = np.arange(x_bias_o.shape[0])
        np.random.shuffle(indices)

        x_bias_shuff = x_bias_o[indices]
        y_shuff = Y[indices]

        n_prime = x_bias_shuff.shape[0] // mini_batch_size
            
        sub_x_shuff = np.array_split(x_bias_shuff, n_prime, axis=0)
        sub_y_shuff = np.array_split(y_shuff, n_prime, axis=0)
        
        # I will leave you to decide how you want to organize the rest of the code, but below is what I used and recommend. Decompose them into private components/functions. 

        ## prep the data: add bias column; randomly shuffle data training set. 

        ## for every iteration:
        #### get a minibatch and use it for:
        ######### forward feeding
        ######### calculate the error of this batch if you want to track/observe the error trend for viewing purpose.
        ######### back propagation to calculate the gradients of all the weights
        ######### use the gradients to update all the weight matrices. 
        for t in range(1, iterations):
         
            if SGD:
                s_indi = random.randrange(0, n_prime)
                d_prime = [sub_x_shuff[s_indi], sub_y_shuff[s_indi]]
            else:
                d_prime = [x_bias_o, Y]  

            # Forward Feed
            self.layers[0].X = d_prime[0] 
            
            for l in range(1, self.L + 1):
                self.layers[l].S = self.layers[l - 1].X @ self.layers[l].W
                if l < self.L:  
                    self.layers[l].X = np.insert(self.layers[l].act(self.layers[l].S), 0, 1, axis=1)
                else:  
                    self.layers[l].X = self.layers[l].act(self.layers[l].S)


            # Error
            E = np.sum((self.layers[self.L].X - np.array(d_prime[1])) ** 2) * (1 / n_prime)
            self.error_list = self.error_list + [E]

            if(t % 1000 == 0):
                os.system('cls')
                print("Training is [ " + format((t/iterations) * 100, '.1f') + "% ] complete.")
                print("Error is: " + str(E))

            # Set delta of output layer
            self.layers[self.L].Delta = 2 * (self.layers[self.L].X - d_prime[1]) * self.layers[self.L].act_de(self.layers[self.L].S)
           
            # Set gradient of output layer
            self.layers[self.L].G = np.einsum('ij,ik -> jk', self.layers[self.L - 1].X, self.layers[self.L].Delta) * (1/n_prime)

            
            for l in range (self.L - 1, 0, -1):
               
                self.layers[l].Delta = self.layers[l].act_de(self.layers[l].S) * (self.layers[l + 1].Delta @ np.transpose(self.layers[l + 1].W[1:, :]))

                self.layers[l].G = np.einsum('ij,ik -> jk', self.layers[l-1].X, self.layers[l].Delta) * (1/n_prime)

            for l in range(1, self.L + 1):
               
                self.layers[l].W = self.layers[l].W - eta * self.layers[l].G

            
              
    def predict(self, X):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column.
            
            return: n x 1 matrix, n is the number of samples, every row is the predicted class id.
            
            Note that the return of this function is NOT the sames as the return of the `NN_Predict` method
            in the lecture slides. In fact, every element in the vector returned by this function is the column
            index of the largest number of each row in the matrix returned by the `NN_Predict` method in the 
            lecture slides.
         '''

        self.layers[0].X = np.insert(X, 0, 1, axis=1)  
    
        for l in range(1, self.L + 1):
            self.layers[l].S = self.layers[l - 1].X @ self.layers[l].W
            if l < self.L:  
                self.layers[l].X = np.insert(self.layers[l].act(self.layers[l].S), 0, 1, axis=1)
            else: 
                self.layers[l].X = self.layers[l].act(self.layers[l].S)
        
        return np.argmax(self.layers[self.L].X, axis=1, keepdims=True) 

    
    def error(self, X, Y):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column. 
               n is the number of samples. 
               d is the number of (non-bias) features of each sample. 
            Y: n x k matrix, the labels of the input n samples. Each row is the label of one sample, 
               where only one entry is 1 and the rest are all 0. 
               Y[i,j]=1 indicates the ith sample belongs to class j.
               k is the number of classes. 
            
            return: the percentage of misclassfied samples
        '''
        pred = self.predict(X)

        labels =  np.argmax(Y, axis=1).reshape(-1, 1)

        miscalc = np.sum(pred != labels)
    
        return (miscalc / X.shape[0]) 
 
