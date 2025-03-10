# Place your EWU ID and name here | ID: 00953041 - Andrew Tucker
## delete the `pass` statement in every function below and add in your own code. 
import numpy as np

# Various math functions, including a collection of activation functions used in NN.

class MyMath:

    def tanh(x):
        ''' tanh function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is tanh of the corresponding element in array x
        '''
        return np.tanh(x)

    
    def tanh_de(x):
        ''' Derivative of the tanh function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is tanh derivative of the corresponding element in array x
        '''

        # tanh'(x) = 1.0 - (tanh(x))^2.0
        return 1.0 - np.power(np.tanh(x), 2)

    
    def logis(x):
        ''' Logistic function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is logistic of 
                    the corresponding element in array x
        '''

        # You can compute logis via tanh as: logix(x) = (tanh(x/2.0) + 1.0) / 2.0
        
        return (np.tanh(x/2.0) + 1.0) / 2.0

    
    def logis_de(x):
        ''' Derivative of the logistic function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is logistic derivative of 
                    the corresponding element in array x
        '''
        #logis'(x) = logis(x) * (1.0-logis(x)). 
        
        return MyMath.logis(x) * (1.0 - MyMath.logis(x))

    
    def iden(x):
        ''' Identity function
            Support vectorized operation
            
            x: an array type of real numbers
            return: the numpy array where every element is the same as
                    the corresponding element in array x
        '''
        return x

    
    def iden_de(x):
        ''' The derivative of the identity function 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array of all ones of the same shape of x.
        '''
        return np.ones_like(x)
        

    def relu(x):
        ''' The ReLU function 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array of the same shape of x, where every element is the max of: zero vs. the corresponding element in x.
        '''
        return np.maximum(0, x)

    
    def relu_de(x):
        ''' The derivative of the ReLU function 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array of the same shape of x, where every element is 1 if the correponding x element is positive; 0, otherwise. 
        '''
        return np.where(x > 0, 1, 0)

    