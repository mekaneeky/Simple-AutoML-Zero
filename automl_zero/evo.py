import numpy as np
from automl_zero.config import *
from automl_zero.ops import pred_OP_dict, setup_OP_dict
from automl_zero.memory import initialize_memory_limited
from automl_zero.mutators import mutate_winner
#from automl_zero.utils import apply_output
from random import randint, sample
from copy import deepcopy
from tqdm import tqdm


def generate_random_gene(OP_dict, 
    MAX_ARG = 2, X_SHAPE = None,y_shape= None ,OP_DEPTH = None,
    CONSTANTS_LOW = -100, CONSTANTS_HIGH= 100,CONSTANTS_MAX =2):

    OP_gene = np.random.randint(0,len(OP_dict), size=(OP_DEPTH,1)).astype(int)
    _, memory_ref_dict = initialize_memory_limited(X_SHAPE, y_shape) # to have correct arg_list_sizes
    arg_locations = np.random.randint(0, len(memory_ref_dict),size=(OP_DEPTH, MAX_ARG)).astype(int)
    output_locations = np.random.randint(0, len(memory_ref_dict) ,size=(OP_DEPTH, 1)).astype(int)
    constants = np.random.uniform(CONSTANTS_LOW,CONSTANTS_HIGH, size=(OP_DEPTH,CONSTANTS_MAX))#gaussian or uniform ??
    
    return OP_gene, arg_locations, output_locations, constants

##TODO Add Initialize Memory
##TODO Classcize this and allow for extentions through inheritance/mixins
def initialize_population(X = None, y = None, population_count = POPULATION_COUNT,\
                          fitness_func=None, \
                          setup_OP_dict = None, SETUP_OP_DEPTH = None , \
                          pred_OP_dict = None, PRED_OP_DEPTH = None, \
                          setup_function = False,
                          learn_function = False):
    genes_list = []

    for _ in tqdm(range(population_count)):
        

        genes_temp = {}
        _, memory_ref_dict = initialize_memory_limited(X.shape, y_true.shape)
        if setup_function:
            # Only one memory for both? 
            gene_setup = generate_random_gene(OP_dict = setup_OP_dict, 
                                              X_SHAPE = X.shape,
                                              y_shape = y_true.shape,
                                              OP_DEPTH = SETUP_OP_DEPTH)
        else:
            gene_setup = None

        gene_pred = generate_random_gene(OP_dict = pred_OP_dict,
                                         X_SHAPE = X.shape, 
                                         y_shape = y_true.shape,
                                         OP_DEPTH = PRED_OP_DEPTH)
        
        """
        if learn_function:
            gene_learn = generate_random_gene(OP_dict = setup_OP_dict, 
                                              X_SHAPE = X.shape,
                                              y_shape = y_true.shape,
                                              OP_DEPTH = LEARN_OP_DEPTH)
        
            genes_temp.append( gene_learn )
        """

        if setup_function:
            resolve_genome(gene = gene_setup,
                           OP_dict=setup_OP_dict,
                          memory_ref_dict= memory_ref_dict,
                          return_result = False)
        
        
        preds = resolve_genome(X = X_arr,y = y_true,
                               gene = gene_pred,
                               memory_ref_dict = memory_ref_dict,
                               OP_dict=pred_OP_dict  )
        fitness = fitness_func(preds, y_true)

        if learn_function:
            ### TODO add training loop and override preds above
            raise NotImplementedError
        else:
            gene_learn = None 

        #TODO test whether we need the memory_arr or just the memory_ref_dict
        #memory_reference = initialize_memory_limited(X.shape, y_true.shape)

        ## too fixed? fuck it 
        genes_list.append( 
            {
            "fitness":fitness, 
            "gene_setup":gene_setup, #should genes be in a list under key gene?
            "gene_pred":gene_pred,
            "gene_learn":gene_learn
            #"memory_reference": memory_reference #do we need to store memory or could it be transient 
            } ) 

    return genes_list


def run_tournament(contestants):
    min_fitness = float("inf")
    for idx in range(len(contestants)):
        if contestants[idx]["fitness"] < min_fitness:
            min_fitness = contestants[idx]["fitness"]
            min_idx = idx
    tournament_winner = contestants[min_idx]
    return tournament_winner


def get_best_fitness(population_list):
    min_fitness = float("inf")
    for meta_gene in population_list:
        if min_fitness > meta_gene["fitness"]:
            min_fitness = meta_gene["fitness"]
    return min_fitness

## TODO this can be vectorized
def run_evolution(X, y, iters = 100000,
                fitness_func = None, \
                population_list = None, 
                N=10):
    """
    There are 2 modes of referencing 
    """
    
    for i in tqdm(range(iters)):   
        
        
        if i%1000 == 0:
            print("BEST FITNESS TO DATE: {}".format(get_best_fitness(population_list)))

        current_memory, memory_ref_dict = initialize_memory_limited(X_shape = X.shape,y_shape = y.shape, scalars=5, vectors=5, matricies=5)

        contestants = sample(population_list, N)
        tournament_winner = run_tournament(contestants)        
        new_metagene = mutate_winner(tournament_winner, len(memory_ref_dict))
        
        #setup function
        resolve_genome(X=X_arr,y=y_true,
                              gene = new_metagene["gene_setup"], \
                              OP_dict = setup_OP_dict, \
                              memory_ref_dict = memory_ref_dict, \
                             return_result = False)
        
        #predict function
        preds = resolve_genome(X=X_arr,y=y_true,
                              gene = new_metagene["gene_pred"], \
                              OP_dict = pred_OP_dict,
                              memory_ref_dict= memory_ref_dict  )# we can supply different OP_dicts to shift meta-levels
        new_metagene["fitness"] = fitness_func(preds, y_true)
        
        ## Append to population list
        population_list.append( new_metagene )
        population_list.pop(0)
        
    return population_list


def resolve_genome(
    X = None,
    y = None,
    gene = None,
    memory_ref_dict = None,
    OP_dict = None,
    return_result = True):
    
    if return_result:
        final_results_arr = np.empty_like(y).astype(float)
    else:
        X = [None] # run once

    genome_OPs, args_locations, output_locations, constants = gene

    
    #for X_idx in range(len(X)):
    #FIXME use mem ref dict
    if return_result:
        memory_ref_dict[0][:] = np.resize(X, memory_ref_dict[0].shape )
    
    for gene_idx in range(0,len(genome_OPs)):
        OP_to_apply, max_args = OP_dict[genome_OPs[gene_idx][0]]

        if max_args == 0:
            op_args = constants[gene_idx]
        else:
            op_args = (memory_ref_dict.get(idx) for idx in args_locations[gene_idx,:max_args])#tuples are quicker?
        
        ## Commit output to memory
        try:
            result = OP_to_apply(*op_args)
        except:
            #failed_op_count += 1
            result = 0#float("nan")
        output_idx = output_locations[gene_idx][0]        
        output_arr = memory_ref_dict[output_idx]
        memory_ref_dict[output_idx][:] = np.resize(result, output_arr.shape)
    
    if return_result:
        return memory_ref_dict[1]
