import numpy as np

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
    
def initialize_memory_limited(X_shape = None,y_shape = None ,scalars=5, vectors=5, matricies=5):
    """
    
    """    

    ## FIXME memory shape should be independent of X,y think GANs, generative models
    vector_limit = scalars + vectors
    matrix_limit = vector_limit + matricies
    memory_arr = np.zeros(shape=(matrix_limit,*X_shape))
    memory_ref_dict = dict()
    
    #0 input, 1 output
    memory_ref_dict[0] = memory_arr[0,0:X_shape[0],0:X_shape[1]]
    memory_ref_dict[1] = memory_arr[1,0:y_shape[0],0:y_shape[1]]
    
    for idx in range(2,scalars):
        memory_ref_dict[idx] = memory_arr[idx,0:1,0:1] 
    for idx in range(scalars,vector_limit):
        memory_ref_dict[idx] = memory_arr[idx,0:X_shape[0],0]       
    for idx in range(vector_limit,matrix_limit):
        memory_ref_dict[idx] = memory_arr[idx,0:X_shape[0],0:X_shape[1]]
        
    return memory_arr, memory_ref_dict
    