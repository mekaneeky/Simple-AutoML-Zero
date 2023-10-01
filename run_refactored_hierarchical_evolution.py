import numpy as np
from automl_zero.config import *
from automl_zero.hierarchical.config import *
from automl_zero.gpt4_refactor.evo import initialize_gene_population,initialize_OP_gene_population, \
     hierarchical_run_gene_evolution, create_OP_gene, hierarchical_run_op_evolution, select_functional_genes_for_op_evaluation,select_op_genes_for_functional_evaluation
from automl_zero.memory import initialize_memory_limited
from automl_zero.fitness import mae
from automl_zero.ops import  setup_OPs, pred_OPs, learn_OPs, numba_test_OP_dict
from time import time
from numba import jit, config

if __name__ == "__main__":

    # Should I also make mutators relative depth aware? 
    MAX_OP_DEPTH = max(SETUP_OP_DEPTH, PRED_OP_DEPTH, LEARN_OP_DEPTH )
    #FIXME fit memory and inference to one example not whole batch at once
    print("Initializing")
    X_arr = np.expand_dims(X_arr, axis=-1)
    y_true = np.expand_dims(y_true, axis=-1)

    memory_ref_dict = initialize_memory_limited(X_shape = X_arr[0].shape, y_shape=y_true[0].shape, 
    scalars = 3, vectors = 3, matricies=3)
    
    print("Generating population")

    #Throwaway initializer gene
    combine_OPs, cached_metalevels = create_OP_gene(METALEVEL_COUNT, NUMBER_OF_META_OPS, PRIOR_LEVEL_OPS)
    #print(cached_metalevels.shape)
    #print(combine_OPs.shape)

        
    

    functional_population_list, functional_fitness_list = initialize_gene_population(
        X = X_arr, y = y_true, \
        memory_ref_dict = memory_ref_dict, \
        fitness_func =mae, \
        #population_count = POPULATION_COUNT,
        #max_OP_depth = MAX_OP_DEPTH,
        #setup_OP_dict = setup_OPs, SETUP_OP_DEPTH = SETUP_OP_DEPTH ,
        #pred_OP_dict = pred_OPs, PRED_OP_DEPTH = PRED_OP_DEPTH, \
        #learn_OP_dict = learn_OPs, LEARN_OP_DEPTH = LEARN_OP_DEPTH, \
        #setup_function = True,
        #learn_function= False,
        #initialization= "random",
        #METALEVEL_COUNT = METALEVEL_COUNT,
        cached_metalevels = cached_metalevels,
       )

        

    OP_population_list, OP_cache_list, OP_fitness_list = initialize_OP_gene_population(X_arr, y_true, memory_ref_dict, mae, 
                    functional_population_list[0], MAX_OP_DEPTH, #throwaway reference gene
                    METALEVEL_COUNT, NUMBER_OF_META_OPS, PRIOR_LEVEL_OPS)

    print("Initialized")
    print("Mutating")
    #import pdb;pdb.set_trace()



    start_time = time()
    functional_OP_genes_selected, functional_OP_cache_selected = select_op_genes_for_functional_evaluation(OP_population_list, OP_cache_list, OP_fitness_list, N=10, count=1)
    functional_OP_genes_selected, functional_OP_cache_selected = functional_OP_genes_selected[0], functional_OP_cache_selected[0]

    functional_population_list, functional_fitness_list, iters = hierarchical_run_gene_evolution(X = X_arr,y = y_true, iters = 100,
                            fitness_func = mae, \
                            population_list=functional_population_list, \
                            fitness_list = functional_fitness_list,
                            N=TOURNAMENT_COUNT,
                            setup_function = True,
                        learn_function= False, 
                        setup_OP_dict = setup_OPs, SETUP_OP_DEPTH = SETUP_OP_DEPTH ,
                    pred_OP_dict = pred_OPs, PRED_OP_DEPTH = PRED_OP_DEPTH, \
                    learn_OP_dict = learn_OPs, LEARN_OP_DEPTH = LEARN_OP_DEPTH, \
                        memory_ref_dict = memory_ref_dict,
                        cached_metalevels = cached_metalevels)
    end_time = time()
    total_time = end_time - start_time
    time_per_seconds = iters/total_time
    print(f"Iters per second: {time_per_seconds}")
    print(f"Time taken: {total_time/60}")

    functional_genes_selected = select_functional_genes_for_op_evaluation(functional_population_list, functional_fitness_list, N = 10, count=1)
    start_time = time()
    OP_population_list, OP_cache_list, OP_fitness_list, iters = hierarchical_run_op_evolution(OP_population_list, OP_cache_list,OP_fitness_list, X_arr, y_true, iters=1000, 
                                  evaluate_steps=10000,reference_metagene= functional_genes_selected, fitness_func=mae, 
                                  N=10, memory_ref_dict=memory_ref_dict, accuracy=False,resolve_depth=MAX_OP_DEPTH, MAX_ARG=2)
    end_time = time()
    total_time = end_time - start_time
    time_per_seconds = iters/total_time
    print(f"Iters per second: {time_per_seconds}")
    print(f"Time taken: {total_time/60}")

#REFERENCE 10,000 models/second/cpu #Max OPS per model 87*(n learn)
# OPS = 870000*n_learn*learn_op_percentage
#PREV max OPS 800/sec
#CURRENT max OPS 2200/sec
#CURRENT max OPS 49251/sec


