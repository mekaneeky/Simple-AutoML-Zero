import numpy as np
from automl_zero.config import *
from automl_zero.evo import initialialize_population, run_evolution, mse_fitness
from automl_zero.ops import  OP_dict_basic

if __name__ == "__main__":

    X_arr = np.array([[10,22, 5000, 300, 999],
                      [1012,242, 50200, 3002, 999],
                      [1013,232, 50400, 3001, 999],
                      [101,212, 52000, 3300, 4999],
                      [1,2, 500, 30, 99],
                      [120,222, 50200, 3200, 9929],
    ]).reshape(-1,5,1)
    y_true = X_arr / 2#X_arr*50**3

    reward_to_gene_list = initialialize_population(X_arr, \
                            y_true = y_true, \
                            fitness_func =mse_fitness, \
                            population_count = POPULATION_COUNT, \
                            op_depth = OP_DEPTH, \
                            memory_size = MEMORY_SIZE , \
                            OP_dict = OP_dict_basic,\
                            max_arg = N_ARGS)

    final_genome = run_evolution(X_arr,y_true,iters = ITERS,\
                                fitness_func = mse_fitness, \
                                reward_to_gene_list=reward_to_gene_list, \
                                N=TOURNAMENT_COUNT, \
                                population = POPULATION_COUNT )
