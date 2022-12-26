import pyglove as pg
import numpy as np
from .config import *
#from numba import jit, njit

def sigmoid(x):
     return 1 /(1 + 1 / np.exp(x))

def leaky_relu(x, alpha):
    return np.where(x > 0, x, x * alpha)

def relu(x):
    x[x<0] =0
    return x

def stable_softmax(x):
    z = x - np.max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator

    return softmax

#FIXME do nothing is identity
def do_nothing(x):
    pass

def mean_axis(x):
    return np.mean(x,axis=0)

def std_axis(x):
    return np.std(x,axis=0)
                                                                                             #OP 56-64

def set_constant_scalar(c,_):#op_56 #One constant
    return c
    
"""
def set_constant_vector(c):#op_57 (i) constant in vector
    
    def _():
        return 
    return _

    
def set_constant_matrix(c):#op_58 (i,j) constant in matrix
"""
    
def uniform_scalar(lower, upper):#op_59 # upper and lower 
    return np.random.uniform(lower, upper, size=(1))#or (1,1)?

    
def uniform_vector(lower, upper):#op_60 vector
    return np.random.uniform(lower, upper, size=(X_SHAPE[0],1)) #or (X_SHAPE[0])?

    
def uniform_matrix(lower, upper):#op_61 matrix
    return np.random.uniform(lower, upper, size=(X_SHAPE[0],X_SHAPE[1]))

    
def gaussian_scalar(mean, std):#op_62 #mean and std
    return np.random.normal(mean,std, size=(1))#or (1,1)?
    
def gaussian_vector(mean, std):#op_63 vector
    return np.random.normal(mean,std,  size=(X_SHAPE[0],1))#or (X_SHAPE[0])?
    

def gaussian_matrix(mean, std):#op_64 matrix
    return np.random.normal(mean,std, size=(X_SHAPE[0],X_SHAPE[1]))


pred_OP_dict = {10:(np.add,2), #basic
           1:(np.subtract,2),
           2:(np.multiply,2),
           3:(np.divide,2),
           4:(np.abs,1), 
           5:(np.reciprocal,1),
           6:(np.sin,1), 
           7:(np.cos,1), 
           8:(np.tan,1),
           9:(np.log,1),
           0:(do_nothing,1)}

setup_OP_dict = {
        7:(set_constant_scalar, 0),
        1:(uniform_scalar, 0),
        2:(uniform_vector, 0),
        3:(uniform_matrix, 0),
        4:(gaussian_scalar, 0),
        5:(gaussian_vector, 0),
        6:(gaussian_matrix, 0),
        0:(do_nothing,1)
        }

learn_OP_dict = {10:(np.add,2), #basic
           1:(np.subtract,2),
           2:(np.multiply,2),
           3:(np.divide,2),
           4:(np.abs,1), 
           5:(np.reciprocal,1),
           6:(np.sin,1), 
           7:(np.cos,1), 
           8:(np.tan,1),
           9:(np.log,1),
           0:(do_nothing,1)}


OP_dict_sizes = { 
    "gene_setup":len(setup_OP_dict),
    "gene_pred":len(pred_OP_dict),
    "gene_learn":len(learn_OP_dict) 

}

"""    
OP_dict_basic = {0:(np.add,2,0,0), 
           1:(np.subtract,2,0,0),
           2:(np.multiply,2,0,0),
           3:(np.divide,2,0,0),
           4:(np.abs,1,0,0), 
           5:(np.reciprocal,1,0,0),
           6:(np.sin,1,0,0), 
           7:(np.cos,1,0,0), 
           8:(np.tan,1,0,0),
           9:(np.log,1,0,0),
           10:(do_nothing,1,0,0)}

OP_dict_int = {11:(np.heaviside,2,0,0),
               12:(np.heaviside,2,1,1),
               13:(np.heaviside,2,2,2),
               14:(np.multiply,2,3,1),
               15:(np.divide,2,1,1),
               16:(np.abs,2,1,1),
               17:(np.add,2,1,1),
               18:(np.divide,2,1,1),
               19:(np.subtract,2,1,1),
               20:(np.multiply,2,1,1),
               21:(np.dot,2,1,0), #FIXME inputs need to be aligned
               22:(np.outer,2,1,2), #FIXME inputs need to be aligned
               23:(np.multiply,2,4,2), #FIXME inputs need to be aligned
               24:(np.reciprocal,2,2,2), #FIXME inputs need to be aligned
               25:(np.dot,2,4,1), #FIXME inputs need to be aligned
               26:(np.transpose,1,2,2),
               27:(np.abs,1,2,2),
                }

OP_dict_int_2 = {28:(np.add,2,2,2),
                 29:(np.divide,2,2,2),
                 30:(np.multiply,2,2,2),
                 31:(np.subtract,2,2,2),
                 32:(np.matmul,2,2,2), #FIXME inputs need to be aligned
                 33:(np.minimum,2,0,0), #ALL mins here are elementwise
                 34:(np.minimum,2,1,1), #ALL mins here are elementwise
                 35:(np.minimum,2,2,2), #ALL mins here are elementwise
                 36:(np.maximum,2,0,0), #ALL mins here are elementwise
                 37:(np.maximum,2,1,1), #ALL mins here are elementwise
                 38:(np.maximum,2,2,2), #ALL mins here are elementwise
                 39:(np.mean,2,1,0),
                 40:(np.mean,2,2,0),
                 41:(mean_axis,2,2,1),
                 42:(std_axis,2,2,1),
                 43:(np.std,2,1,0),
                 44:(np.std,2,2,0)

                }
"""