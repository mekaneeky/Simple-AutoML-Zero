import numpy as np
from automl_zero.config import *
from automl_zero.evo import initialize_population, run_evolution
from automl_zero.fitness import mae
from automl_zero.ops import  pred_OP_dict, setup_OP_dict

if __name__ == "__main__":

    SETUP_OP_DEPTH = 5
    PRED_OP_DEPTH = 10
    #X_arr*50**3

    #   (X = None, y = None, population_count = POPULATION_COUNT,\
    #   fitness_func=None, op_depth = None,
    #   memory_arr = None, memory_ref_dict = None, \
    #   setup_OP_dict = None, SETUP_OP_DEPTH = None ,
    #   pred_OP_dict = None, PRED_OP_DEPTH = None, \
    #   setup_function = False)
    population_list = initialize_population(
        X = X_arr, y = y_true, \
        fitness_func =mae, \
        population_count = POPULATION_COUNT, \
        setup_OP_dict = setup_OP_dict, SETUP_OP_DEPTH = SETUP_OP_DEPTH ,
        pred_OP_dict = pred_OP_dict, PRED_OP_DEPTH = PRED_OP_DEPTH, \
        setup_function = True)

    #run_setup_evolution(X, y, iters = 100000,
    #            fitness_func = None, \
    #            reward_to_gene_list = None, 
    #            N=10, population = POPULATION_COUNT ):
    final_genome = run_evolution(X = X_arr,y = y_true,
                            iters = ITERS, fitness_func = mae, \
                            population_list=population_list, \
                            N=TOURNAMENT_COUNT )

#PREV max OPS 800/sec