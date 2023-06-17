import numpy as np 

from numba import njit
from numba.typed import Dict, List
from numpy.random import choice, random
from copy import deepcopy,copy
from automl_zero.ops import LOW_SETUP_OPS,UNIQUE_SETUP_OPS,LOW_PRED_OPS,UNIQUE_PRED_OPS,LOW_LEARN_OPS,UNIQUE_LEARN_OPS
from automl_zero.hierarchical.config import *
from automl_zero.hierarchical.op_combination import create_OP_population, get_ops_from_meta_level, _generate_cached_metalevels
from numba import njit

#from automl_zero.memory import initialize_memory_limited

## TODO use globals ?
#@njit(cache=True)
def _mutate_all(winner,gene_to_mutate, memory_dict_len, MAX_ARG=2):
    print("mutate_all")
    #CONSTANTS_LOW = -100,CONSTANTS_HIGH = 100
    genome_OPs, metalevel_gene, args_locations, output_locations, constants = winner[:,0:1], winner[:,1:2], winner[:,2:2+MAX_ARG], winner[:,2+MAX_ARG:3+MAX_ARG],winner[:,3+MAX_ARG:]

    metalevel_gene = np.random.randint(0,METALEVEL_COUNT, size=(metalevel_gene.shape[0],metalevel_gene.shape[1])).astype(np.int8)

    print(f"NUMBER OF META OPS: {NUMBER_OF_META_OPS}")    
    genome_OPs = np.random.randint(0,NUMBER_OF_META_OPS, size=(genome_OPs.shape[0],genome_OPs.shape[1])).astype(np.int8)




    args_locations = np.random.randint(0, memory_dict_len,size=(args_locations.shape)).astype(np.float64)
    output_locations = np.random.randint(0, memory_dict_len ,size=(output_locations.shape[0],1)).astype(np.float64)
    constants = np.random.uniform(-100,100, size=constants.shape)

    return np.hstack((genome_OPs, metalevel_gene, args_locations, output_locations,constants))

#@njit(cache=True)
def _mutate_add_or_remove_one_instruction(winner, gene_to_mutate,  memory_dict_len, MAX_ARG=2):
    print("mutate one")
    #CONSTANTS_LOW = -100,CONSTANTS_HIGH = 100
    genome_OPs, metalevel_gene, args_locations, output_locations, constants = winner[:,0:1], winner[:,1:2], winner[:,2:2+MAX_ARG], winner[:,2+MAX_ARG:3+MAX_ARG],winner[:,3+MAX_ARG:]

    instruction_idx = np.random.randint(0,genome_OPs.shape[0], size=(1) )[0]


    metalevel_gene[instruction_idx:instruction_idx+1] = np.random.randint(0,METALEVEL_COUNT, size=(1)).astype(np.int8)
    genome_OPs[instruction_idx:instruction_idx+1] = np.random.randint(0,NUMBER_OF_META_OPS, size=(1)).astype(np.int8)
                

    args_locations[instruction_idx] = np.random.randint(0, memory_dict_len,size=args_locations[instruction_idx].shape).astype(np.int8)

    output_locations[instruction_idx:instruction_idx+1] = np.random.randint(0, memory_dict_len,size=(1)).astype(np.int8)
    constants[instruction_idx:instruction_idx+1] = np.random.uniform(-100,100, size=(1))
    
    return np.hstack((genome_OPs,metalevel_gene ,args_locations, output_locations, constants))

#@njit(cache=True)
def _mutate_one_argument(winner, gene_to_mutate, memory_dict_len, MAX_ARG=2):
    print("mutate one")
    genome_OPs, metalevel_gene, args_locations, output_locations, constants = winner[:,0:1], winner[:,1:2], winner[:,2:2+MAX_ARG], winner[:,2+MAX_ARG:3+MAX_ARG],winner[:,3+MAX_ARG:]

    instruction_idx = np.random.randint(0,genome_OPs.shape[0], size=(1))[0]

    if np.random.random() > 0.5:
        argument_idx = np.random.randint(0,args_locations[instruction_idx].shape[0], size=(1))[0]
        args_locations[instruction_idx,argument_idx:argument_idx+1] = np.random.randint(0, memory_dict_len,size=(1,1)).astype(np.float64)
    else:
        output_locations[instruction_idx:instruction_idx+1] = np.random.randint(0, memory_dict_len,size=(1)).astype(np.float64)
        
    #(fitness,(OP_gene, memory_arr, arg_locations, output_locations )
    return np.hstack((genome_OPs,metalevel_gene ,args_locations, output_locations, constants))

#@njit(cache=True)
def mutate_winner(winner_metagene, memory_dict_len):
    ## namespace
    ## Metalearners that learn across meta-levels 
    ## Should this be a hyperparameter as well too?
    ## The whole metagene should be copied

    ## TODO (hypothetical)
    ## Otherwise we end up with the scenario where when a gene is modified it will change 
    ## across generations !! wait it won't miscalculation
    ## However, more importantly can we do this 
    ## and only allow a mutation that leads to better overall ancestral performance? 

    ##FIXME sample from range of metagene.keys()
    """
    if winner_metagene["gene_setup"] != None and winner_metagene["gene_learn"] != None:
        gene_to_mutate = choice(np.array(["gene_setup", "gene_pred", "gene_learn"]))
    elif winner_metagene["gene_setup"] != None:
        gene_to_mutate = choice(np.array(["gene_setup", "gene_pred"]))
    else:
        gene_to_mutate = "gene_pred"
    """
    gene_to_mutate_idx = choice(len(winner_metagene))
    #gene_to_mutate = winner_metagene.keys()[gene_to_mutate]
    
    dice_roll = random()
    if dice_roll < 0.3333: 
        mutation_function = _mutate_one_argument
    elif dice_roll < 0.6666:
        mutation_function = _mutate_all
    else:
        mutation_function = _mutate_add_or_remove_one_instruction
    
    #new_metagene = deepcopy(winner_metagene) #switching to copy boosted by 300-400 iters 
    new_metagene = Dict()
    counter = 0 
    for key in winner_metagene:

        new_metagene[key] = winner_metagene[key]

        if counter == gene_to_mutate_idx:
            gene_to_mutate = key
        counter += 1
    # Should we have a state of all functions 
    # in a dictionary where 42:function_for_all()

    #import pdb;pdb.set_trace()
    new_metagene[gene_to_mutate] = mutation_function(new_metagene[gene_to_mutate].copy(), gene_to_mutate, memory_dict_len)
    return new_metagene

# Complete this function
#@njit(cache=True)
def mutate_combination_single(combine_OPs, NUMBER_OF_BASE_OPS = None):
    #combine_OPs = np.random.randint(0, NUMBER_OF_OPS , size=(METALEVEL_COUNT, NUMBER_OF_OPS,PRIOR_LEVEL_OPS )).astype(np.float64)   
    METALEVEL_COUNT, NUMBER_OF_OPS,PRIOR_LEVEL_OPS = combine_OPs.shape
    metalevel_idx = np.randint(0, METALEVEL_COUNT, size=(1))[0]
    op_idx = np.randint(0, NUMBER_OF_OPS, size=(1))[0]
    new_op_idx = np.randint(0, PRIOR_LEVEL_OPS, size=(1))[0]
    new_op_value = np.randint(0, NUMBER_OF_OPS, size=(1))[0]


    new_op_value = np.randint(0, NUMBER_OF_OPS, size=(1))[0]

    combine_OPs[metalevel_idx,op_idx, new_op_idx] = new_op_value
    

    return combine_OPs

#@njit(cache=True)
def mutate_combination_all_ops(combine_OPs, NUMBER_OF_BASE_OPS = None):
    #combine_OPs = np.random.randint(0, NUMBER_OF_OPS , size=(METALEVEL_COUNT, NUMBER_OF_OPS,PRIOR_LEVEL_OPS )).astype(np.float64)   
    METALEVEL_COUNT, NUMBER_OF_OPS,PRIOR_LEVEL_OPS = combine_OPs.shape
    metalevel_idx = np.randint(0, METALEVEL_COUNT, size=(1))[0]
    op_idx = np.randint(0, NUMBER_OF_OPS, size=(1))[0]
    new_ops = np.randint(0, NUMBER_OF_OPS, size=(PRIOR_LEVEL_OPS))[0]

    combine_OPs[metalevel_idx,op_idx, :] = new_ops
    
    return combine_OPs

#@njit(cache=True)
def mutate_combination_all(combine_OPs, NUMBER_OF_BASE_OPS = None):
    #combine_OPs = np.random.randint(0, NUMBER_OF_OPS , size=(METALEVEL_COUNT, NUMBER_OF_OPS,PRIOR_LEVEL_OPS )).astype(np.float64)   
    #_COUNT_OF_GENES ,METALEVEL_COUNT, NUMBER_OF_OPS,PRIOR_LEVEL_OPS

    METALEVEL_COUNT, NUMBER_OF_META_OPS,PRIOR_LEVEL_OPS = combine_OPs.shape

    
    combine_OPs = np.random.randint(0, NUMBER_OF_META_OPS , size=(METALEVEL_COUNT, NUMBER_OF_META_OPS,PRIOR_LEVEL_OPS )).astype(np.float64)
    #Adding the base ops as a separate layer
    #combine_OPs[0,:,:] = np.random.randint(0, NUMBER_OF_BASE_OPS , size=(1, NUMBER_OF_META_OPS,PRIOR_LEVEL_OPS )).astype(np.float64)

    return combine_OPs

#@njit(cache=True)
def mutate_combination_winner(winner_combine_OPs, winner_cached_gene, NUMBER_OF_BASE_OPS = None):
    ## namespace
    ## Metalearners that learn across meta-levels 
    ## Should this be a hyperparameter as well too?
    ## The whole metagene should be copied
    #NUMBER_OF_BASE_OPS: needs to be a dict sharing the keys of winner_combine_OPs and winner_cached_gene
    #each value would be the count of ops in each level

    ## TODO (hypothetical)
    ## Otherwise we end up with the scenario where when a gene is modified it will change 
    ## across generations !! wait it won't miscalculation
    ## However, more importantly can we do this 
    ## and only allow a mutation that leads to better overall ancestral performance? 

    ##FIXME sample from range of metagene.keys()
    gene_to_mutate_idx = choice(len(winner_combine_OPs))

    dice_roll = random()
    if dice_roll < 0.3333: 
        mutation_function = mutate_combination_single
    elif dice_roll < 0.6666:
        mutation_function = mutate_combination_all_ops
    else:
        mutation_function = mutate_combination_all

    new_op_metagene = Dict()
    new_cached_gene = Dict()
    counter = 0 
    for key in winner_combine_OPs:

        new_op_metagene[key] = winner_combine_OPs[key].copy() #FIXME slow?
        new_cached_gene[key] = winner_cached_gene[key].copy() #FIXME slow?

        if counter == gene_to_mutate_idx:
            gene_to_mutate = key
        counter += 1

    new_op_metagene[gene_to_mutate] = mutation_function(winner_combine_OPs.copy(), NUMBER_OF_BASE_OPS)

    new_cached_gene[gene_to_mutate] =  _generate_cached_metalevels(new_op_metagene[gene_to_mutate])
    return new_op_metagene, new_cached_gene
   