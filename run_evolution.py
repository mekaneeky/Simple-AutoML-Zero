import numpy as np
from automl_zero.config import *
from automl_zero.evo import initialize_population, run_evolution
from automl_zero.memory import initialize_memory_limited
from automl_zero.fitness import mae
from automl_zero.ops import  pred_OP_dict, setup_OP_dict, learn_OP_dict
#import cProfile 

if __name__ == "__main__":

    SETUP_OP_DEPTH = 2
    PRED_OP_DEPTH = 1
    LEARN_OP_DEPTH = 3
    

    memory_ref_dict = initialize_memory_limited(X_shape = X_arr.shape, y_shape=y_true.shape, 
    scalars = 1, vectors = 1, matricies=1)

    population_list = initialize_population(
        X = X_arr, y = y_true, \
        memory_ref_dict = memory_ref_dict, \
        fitness_func =mae, \
        population_count = POPULATION_COUNT, \
        setup_OP_dict = setup_OP_dict, SETUP_OP_DEPTH = SETUP_OP_DEPTH ,
        pred_OP_dict = pred_OP_dict, PRED_OP_DEPTH = PRED_OP_DEPTH, \
        learn_OP_dict = learn_OP_dict, LEARN_OP_DEPTH = LEARN_OP_DEPTH, \
        setup_function = True,
        learn_function= False,
        initialization= "zeros")

    
    run_evolution(X = X_arr,y = y_true, iters = 1000000,#ITERS,
                            fitness_func = mae, \
                            population_list=population_list, \
                            N=TOURNAMENT_COUNT,
                            setup_function = True,
                        learn_function= False, )

#PREV max OPS 800/sec
#CURRENT max OPS 2200/sec