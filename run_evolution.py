import numpy as np
from automl_zero.config import *
from automl_zero.evo import initialize_population, run_evolution
from automl_zero.memory import initialize_memory_limited
from automl_zero.fitness import mae
from automl_zero.ops import  setup_OPs, pred_OPs, learn_OPs, numba_test_OP_dict
from time import time
#import cProfile 

if __name__ == "__main__":

    
    SETUP_OP_DEPTH = 5
    PRED_OP_DEPTH = 5
    LEARN_OP_DEPTH = 5
    # Should I also make mutators relative depth aware? 
    MAX_OP_DEPTH = max(SETUP_OP_DEPTH, PRED_OP_DEPTH, LEARN_OP_DEPTH )
    #FIXME fit memory and inference to one example not whole batch at once
    memory_ref_dict = initialize_memory_limited(X_shape = (5,1), y_shape=(5,1), 
    scalars = 3, vectors = 3, matricies=3)

    population_list, fitness_list = initialize_population(
        X = X_arr, y = y_true, \
        memory_ref_dict = memory_ref_dict, \
        fitness_func =mae, \
        population_count = 1000,#POPULATION_COUNT, \
        max_OP_depth = MAX_OP_DEPTH,
        setup_OP_dict = setup_OPs, SETUP_OP_DEPTH = SETUP_OP_DEPTH ,
        pred_OP_dict = pred_OPs, PRED_OP_DEPTH = PRED_OP_DEPTH, \
        learn_OP_dict = learn_OPs, LEARN_OP_DEPTH = LEARN_OP_DEPTH, \
        setup_function = True,
        learn_function= True,
        initialization= "zeros")

    start_time = time()
    pop_list, fit_list, iters = run_evolution(X = X_arr,y = y_true, iters = 1000000,#ITERS,
                            fitness_func = mae, \
                            population_list=population_list, \
                            fitness_list = fitness_list,
                            N=TOURNAMENT_COUNT,
                            setup_function = True,
                        learn_function= True, 
                        setup_OP_dict = setup_OPs, SETUP_OP_DEPTH = SETUP_OP_DEPTH ,
                    pred_OP_dict = pred_OPs, PRED_OP_DEPTH = PRED_OP_DEPTH, \
                    learn_OP_dict = learn_OPs, LEARN_OP_DEPTH = LEARN_OP_DEPTH, \
                        memory_ref_dict = memory_ref_dict)
    end_time = time()
    total_time = end_time - start_time
    time_per_seconds = iters/total_time
    print(f"Iters per second: {time_per_seconds}")
    print(f"Time taken: {total_time/60}")

#PREV max OPS 800/sec
#CURRENT max OPS 2200/sec
#CURRENT max OPS 49251/sec