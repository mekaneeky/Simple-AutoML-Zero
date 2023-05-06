import numpy as np
from numba import njit, jit, typed
from automl_zero.config import X_arr, y_true

#@njit(cache=True)
def initialize_memory_free(memory_shape = (30,10,10), mem_type = "normal", mean=0, loc=1, start=-2, end=2):
    """
    
    """    
    if mem_type == "normal":
        init_memory = np.random.normal(loc=mean , scale=loc, size=memory_shape )
        return init_memory
    
    if mem_type == "uniform":
        init_memory = np.random.randint(start , end, size=memory_shape )
        return init_memory

    if mem_type == "zero":
        init_memory = np.zeros(shape=memory_shape)
        return init_memory

#@njit(cache=True)
def initialize_memory_limited(X_shape = X_arr[0].shape,y_shape = y_true[0].shape ,scalars=5, vectors=5, matricies=5):
    #print("initialize memory")
    """
    
    """    

    ## FIXME memory shape should be independent of X,y think GANs, generative models
    scalar_limit = scalars + 3
    vector_limit = scalar_limit + vectors
    matrix_limit = vector_limit + matricies
    memory_arr = np.zeros(shape=(matrix_limit,*X_shape),dtype=np.float64)
    memory_ref_dict = typed.Dict()
    
    #0 input, 1 output, 2 true_y
    memory_ref_dict[0] = memory_arr[0:1,0:X_shape[0],0:X_shape[1]] #input 
    memory_ref_dict[1] = memory_arr[1:2,0:y_shape[0],0:y_shape[1]] #output
    memory_ref_dict[2] = memory_arr[2:3,0:y_shape[0],0:y_shape[1]] #true_label #TODO test leakage

    for idx in range(3,scalar_limit):
        memory_ref_dict[idx] = memory_arr[idx:idx+1,0:1,0:1] 
    for idx in range(scalar_limit,vector_limit):
        memory_ref_dict[idx] = memory_arr[idx:idx+1,0:X_shape[0]]       
    for idx in range(vector_limit,matrix_limit):
        memory_ref_dict[idx] = memory_arr[idx:idx+1,0:X_shape[0],0:X_shape[1]]
        
    return memory_ref_dict
    