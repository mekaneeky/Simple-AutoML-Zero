import numpy as np
from .config import *
from numba import njit,cfunc, typed
from numba.typed import List, Dict
## OP 0: Do Nothing
#FIXME do nothing is identity
#@njit(cache=True)
def do_nothing(*args):
    return 0

#OP 1: Add Scalars
#@njit(cache=True)
def add_scalar(*args):
    return np.add(args[0],args[1])

#OP 2: Subtract Scalars
#@njit(cache=True)
def sub_scalar(*args):
    return np.diff(args[0],args[1])

#OP 3: Subtract Scalars
#@njit(cache=True)
def multiply_scalar(*args):
    return np.multiply(args[0],args[1])

#OP 4: Subtract Scalars
#@njit(cache=True)
def divide_scalar(*args):
    return np.divide(args[0],args[1])

#OP 5: |ABS| Scalars
#@njit(cache=True)
def abs_scalar(*args):
    return np.abs(args[0])

#OP 6: 1/X Scalars
#@njit(cache=True)
def reciprocal_scalar(*args):
    return np.reciprocal(args[0])

#OP 7: sin Scalars
#@njit(cache=True)
def sin_scalar(*args):
    return np.sin(args[0])

#OP 8: كس Scalars
#@njit(cache=True)
def cos_scalar(*args):
    return np.cos(args[0])

#OP 9: tan Scalars
#@njit(cache=True)
def tan_scalar(*args):
    return np.tan(args[0])

#OP 10: arcsin Scalars
#@njit(cache=True)
def arcsin_scalar(*args):
    return np.sin(args[0])

#OP 11: arcكس Scalars
#@njit(cache=True)
def arccos_scalar(*args):
    return np.cos(args[0])

#OP 12: arctan Scalars
#@njit(cache=True)
def arctan_scalar(*args):
    return np.tan(args[0])

##@njit(cache=True)
#def resolve_OP(op_number = 0.0):
    if op_number == 0:
        return do_nothing
    elif op_number == 1:
        return add_scalar
    elif op_number == 2:
        return sub_scalar
    elif op_number == 3:
        return multiply_scalar
    elif op_number == 4:
        return divide_scalar
    elif op_number == 5:
        return abs_scalar
    elif op_number == 6:
        return reciprocal_scalar
    elif op_number == 7:
        return sin_scalar
    elif op_number == 8:
        return cos_scalar
    elif op_number == 9:
        return tan_scalar
    elif op_number == 10:
        return arcsin_scalar
    elif op_number == 11:
        return arccos_scalar
    elif op_number == 12:
        return arctan_scalar
    """
    elif op_number == 13:
        
    elif op_number == 14:

    elif op_number == 15:

    elif op_number == 16:

    elif op_number == 17:
    
    elif op_number == 18:

    elif op_number == 19:

    elif op_number == 20:

    elif op_number == 21:

    elif op_number == 22:

    elif op_number == 23:

    elif op_number == 24:

    elif op_number == 25:

    elif op_number == 26:

    elif op_number == 27:

    elif op_number == 28:

    elif op_number == 29:

    elif op_number == 30:

    elif op_number == 31:

    elif op_number == 32:
    """

def sigmoid(*args):
     return 1 /(1 + 1 / np.exp(x))

def leaky_relu(x, alpha):
    return np.where(x > 0, x, x * alpha)

def relu(*args):
    x[x<0] =0
    return x

def stable_softmax(*args):
    z = x - np.max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator

    return softmax


def mean_axis(*args):
    return np.mean(x,axis=0)

def std_axis(*args):
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

# How to reimplement args vs consts dilemma? can we pass 2 sets of args to f(x)
# We can do it on a setup gene level, but this would impede the pred and learn genes no?
numba_test_OP_dict = {
       0:add_scalar,
       1:divide_scalar,
       2:sub_scalar,
       3:multiply_scalar,
       4:reciprocal_scalar,
       5:abs_scalar 
    }

# TODO tests for scalar, vector and matrix OPS
# TODO tests for cross OP type operations ? What to do then ?


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
pred_OPs = np.random.randint(0,12,size=(11))#List(pred_OP_dict.values())
PRED_OP_NUMBER = 11

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
setup_OPs = np.random.randint(0,10,size=(11))
SETUP_OP_NUMBER = 8

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
learn_OPs = np.random.randint(0,12,size=(11))
LEARN_OP_NUMBER = 11


OP_dict_sizes = Dict() 
#OP_dict_sizes["gene_setup"] = len(setup_OP_dict)
LOW_SETUP_OPS = 0
UNIQUE_SETUP_OPS = len(setup_OP_dict)
#OP_dict_sizes["gene_pred"] = len(pred_OP_dict)
LOW_PRED_OPS = 0
UNIQUE_PRED_OPS = len(pred_OP_dict)
#OP_dict_sizes["gene_learn"] = len(learn_OP_dict) 
LOW_LEARN_OPS = 0
UNIQUE_LEARN_OPS = len(learn_OP_dict)


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