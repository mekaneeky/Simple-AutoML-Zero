import time
import numpy as np
from automl_zero.config import *
#FIXME runtime OP lists are in the config file
from automl_zero.ops import pred_OP_dict, setup_OP_dict, learn_OP_dict
from automl_zero.memory import initialize_memory_limited
from automl_zero.mutators import mutate_winner
#from automl_zero.utils import apply_output
from random import randint, sample
from copy import copy
from tqdm import tqdm
from numba import jit


def generate_random_gene(OP_dict, 
    MAX_ARG = 2, X_SHAPE = None,y_shape= None ,OP_DEPTH = None,
    CONSTANTS_LOW = -50, CONSTANTS_HIGH= 50,CONSTANTS_MAX =2,
    initialization = "random"):

    if initialization == "random":
        OP_gene = np.random.randint(0,len(OP_dict), size=(OP_DEPTH,1)).astype(int)
        memory_ref_dict = initialize_memory_limited(X_SHAPE, y_shape) # to have correct arg_list_sizes
        arg_locations = np.random.randint(0, len(memory_ref_dict),size=(OP_DEPTH, MAX_ARG)).astype(int)
        output_locations = np.random.randint(0, len(memory_ref_dict) ,size=(OP_DEPTH, 1)).astype(int)
        constants = np.random.uniform(CONSTANTS_LOW,CONSTANTS_HIGH, size=(OP_DEPTH,CONSTANTS_MAX))#gaussian or uniform ??
    
    elif initialization == "zeros":
        OP_gene = np.zeros(shape=(OP_DEPTH,1)).astype(int)
        memory_ref_dict = initialize_memory_limited(X_SHAPE, y_shape) # to have correct arg_list_sizes
        arg_locations = np.zeros(shape=(OP_DEPTH, MAX_ARG)).astype(int)
        output_locations = np.zeros(shape=(OP_DEPTH, 1)).astype(int)
        constants = np.zeros(shape=(OP_DEPTH,CONSTANTS_MAX))#gaussian or uniform ??        

    return OP_gene, arg_locations, output_locations, constants

##TODO Classcize this and allow for extentions through inheritance/mixins
def initialize_population(X = None, y = None, population_count = POPULATION_COUNT,\
                          memory_ref_dict = None, \
                          fitness_func=None, \
                          setup_OP_dict = None, SETUP_OP_DEPTH = None , \
                          pred_OP_dict = None, PRED_OP_DEPTH = None, \
                          learn_OP_dict = None, LEARN_OP_DEPTH = None, \
                          setup_function = False,
                          learn_function = False,
                          initialization= "zeros"):
    genes_list = []

    for _ in tqdm(range(population_count)):
        

        if setup_function:
            # Only one memory for both? 
            gene_setup = generate_random_gene(OP_dict = setup_OP_dict, 
                                              X_SHAPE = X.shape,
                                              y_shape = y_true.shape,
                                              OP_DEPTH = SETUP_OP_DEPTH,
                                              initialization = initialization)
        else:
            gene_setup = None

        gene_pred = generate_random_gene(OP_dict = pred_OP_dict,
                                         X_SHAPE = X.shape, 
                                         y_shape = y_true.shape,
                                         OP_DEPTH = PRED_OP_DEPTH,
                                         initialization = initialization)
        
        
        if learn_function:
            gene_learn = generate_random_gene(OP_dict = setup_OP_dict, 
                                              X_SHAPE = X.shape,
                                              y_shape = y_true.shape,
                                              OP_DEPTH = LEARN_OP_DEPTH,
                                              initialization = initialization)
        else:
            gene_learn = None
        
        fitness = 0
        
        for X_val, y_val in zip(X,y):            
            
            #remove previous y used in learn to avoid leakage if repeat value/single example
            memory_ref_dict[2][:] = np.zeros(shape=memory_ref_dict[2].shape )
            
            #setup function
            if setup_function:
                resolve_genome(X=X_val,y=y_val,
                                    gene = gene_setup, \
                                    OP_dict = setup_OP_dict, \
                                    memory_ref_dict = memory_ref_dict)
            
            #predict function
            memory_ref_dict[0][:] = np.resize(X, memory_ref_dict[0].shape )
            preds = resolve_genome(X=X_val,y=y_val,
                                gene = gene_pred, \
                                OP_dict = pred_OP_dict,
                                memory_ref_dict= memory_ref_dict  )# we can supply different OP_dicts to shift meta-levels
            
            #fitness is for a single pred here not all? 
            # we do a sum ? 
            fitness += fitness_func(preds, y_true)
            
            if learn_function:
                # Add real y to allow for learning 
                memory_ref_dict[2][:] = np.resize(y_val, memory_ref_dict[2].shape )

                resolve_genome(X=X_val,y=y_val,
                        gene = gene_learn, \
                        OP_dict = learn_OP_dict,
                        memory_ref_dict= memory_ref_dict  )# we can supply different OP_dicts to shift meta-levels

                
        genes_list.append( 
            {
            "fitness":fitness, 
            "gene_setup":gene_setup, #should genes be in a list under key gene?
            "gene_pred":gene_pred,
            "gene_learn":gene_learn
            } ) 

    return genes_list


def run_tournament(contestants):
    min_fitness = float("inf")
    for idx in range(len(contestants)):
        if contestants[idx]["fitness"] <= min_fitness:
            min_fitness = contestants[idx]["fitness"]
            min_idx = idx
    
    tournament_winner = contestants[min_idx]
    
    return tournament_winner

def get_best_fitness(population_list):
    min_fitness = float("inf")
    for meta_gene in population_list:
        if min_fitness >= meta_gene["fitness"]:
            min_fitness = meta_gene["fitness"]
    return min_fitness

## TODO this can be vectorized
#@jit(nopython=True)
def run_evolution(X, y, iters = 100000,
                fitness_func = None, \
                population_list = None, 
                setup_function = True,
                setup_OP_dict = setup_OP_dict,
                pred_OP_dict = pred_OP_dict,
                learn_function = True,
                learn_OP_dict = learn_OP_dict,
                N=10):
    """
    There are 2 modes of referencing 
    """
    
    memory_ref_dict = initialize_memory_limited(X_shape = X.shape,y_shape = y.shape, scalars=5, vectors=5, matricies=5)

    #start_time = time.time()
    for i in range(iters):
        
        if i%10000 == 0:
            best_fitness = get_best_fitness(population_list)
            print("BEST FITNESS TO DATE: {}".format(best_fitness))
            if best_fitness == 0:
                return population_list
        
        contestants = sample(population_list, N)
        tournament_winner = run_tournament(contestants)        
        new_metagene = mutate_winner(tournament_winner, len(memory_ref_dict))
        ## FIXME How can this cause issues? Leak zeros inappropriatly?
        new_metagene["fitness"] = 0

        #TODO add in f(x)? profile
        #TODO profile this zip vs range(len())
        
        memory_ref_dict[2][:] = np.zeros(shape=memory_ref_dict[2].shape )
        #setup function
        if setup_function:
            resolve_genome(X=None,y=None,
                                gene = new_metagene["gene_setup"], \
                                OP_dict = setup_OP_dict, \
                                memory_ref_dict = memory_ref_dict)
            
        for X_val, y_val in zip(X,y):        
            
            
            
            #remove previous y used in learn to avoid leakage if repeat value/single example
            memory_ref_dict[2][:] = np.zeros(shape=memory_ref_dict[2].shape )
            
            
            #predict function
            memory_ref_dict[0][:] = np.resize(X, memory_ref_dict[0].shape )
            preds = resolve_genome(X=X_val,y=y_val,
                                gene = new_metagene["gene_pred"], \
                                OP_dict = pred_OP_dict,
                                memory_ref_dict= memory_ref_dict  )# we can supply different OP_dicts to shift meta-levels
            
            #fitness is for a single pred here not all? 
            # we do a sum ? 
            new_metagene["fitness"] += fitness_func(preds, y_true)
            
            if learn_function:
                # Add real y to allow for learning 
                memory_ref_dict[2][:] = np.resize(y_val, memory_ref_dict[2].shape )

                resolve_genome(X=X_val,y=y_val,
                        gene = new_metagene["gene_learn"], \
                        OP_dict = learn_OP_dict,
                        memory_ref_dict= memory_ref_dict  )# we can supply different OP_dicts to shift meta-levels

        ## Append to population list
        population_list.append( new_metagene )
        population_list.pop(0)
    #end_time = time.time()
    #seconds_spent = end_time-start_time
    #iters_per_second = iters/seconds_spent
    #print(f"ITERS PER SECOND: {iters_per_second}" )
    return population_list


## This should be context agnostic? 
## While memory_ref_dict carries the context instead? 
def resolve_genome(
    X = None,
    y = None,
    gene = None,
    memory_ref_dict = None,
    OP_dict = None):

    final_results_arr = np.empty_like(y).astype(float)
    genome_OPs, args_locations, output_locations, constants = gene
    
    for gene_idx in range(0,len(genome_OPs)):
        OP_to_apply, max_args = OP_dict[genome_OPs[gene_idx][0]]

        if max_args == 0:
            op_args = constants[gene_idx]

        else:
            op_args = (memory_ref_dict.get(idx) for idx in args_locations[gene_idx,:max_args])#tuples are quicker?
        
        try:
            result = OP_to_apply(*op_args)
        except:
            result = 0#float("nan")

        output_idx = output_locations[gene_idx][0]        
        output_arr = memory_ref_dict[output_idx]
        

        try: 
            memory_ref_dict[output_idx][:] = result#np.resize(result, output_arr.shape)
        except:
            #memory_ref_dict[output_idx][:] = np.resize(result, output_arr.shape)
            memory_ref_dict[output_idx][:] = np.zeros(shape= memory_ref_dict[output_idx].shape)
    
    return memory_ref_dict[1]
