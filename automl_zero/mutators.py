import numpy as np 

from numba import njit
from numba.typed import Dict, List
from numpy.random import choice, random
from copy import deepcopy,copy
from automl_zero.ops import LOW_SETUP_OPS,UNIQUE_SETUP_OPS,LOW_PRED_OPS,UNIQUE_PRED_OPS,LOW_LEARN_OPS,UNIQUE_LEARN_OPS
#from automl_zero.memory import initialize_memory_limited

## TODO use globals ?
@njit(cache=True)
def _mutate_all(winner,gene_to_mutate, memory_dict_len):

    #CONSTANTS_LOW = -100,CONSTANTS_HIGH = 100
    OP_gene, arg_locations, output_locations, constants = winner[:,0:1], winner[:,1:3], winner[:,3:4], winner[:,4:6]
    if gene_to_mutate == "gene_setup":
        OP_gene = np.random.randint(LOW_SETUP_OPS,UNIQUE_SETUP_OPS, size=(OP_gene.shape[0],1)).astype(np.float64)
    elif gene_to_mutate == "gene_pred":
        OP_gene = np.random.randint(LOW_PRED_OPS,UNIQUE_PRED_OPS, size=(OP_gene.shape[0],1)).astype(np.float64)
    else:
        OP_gene = np.random.randint(LOW_LEARN_OPS,UNIQUE_LEARN_OPS, size=(OP_gene.shape[0],1)).astype(np.float64)
    arg_locations = np.random.randint(0, memory_dict_len,size=(arg_locations.shape)).astype(np.float64)
    output_locations = np.random.randint(0, memory_dict_len ,size=(output_locations.shape[0],1)).astype(np.float64)
    constants = np.random.uniform(-100,100, size=constants.shape)

    return np.hstack((OP_gene, arg_locations, output_locations,constants))

@njit(cache=True)
def _mutate_add_or_remove_one_instruction(winner, gene_to_mutate,  memory_dict_len):

    #CONSTANTS_LOW = -100,CONSTANTS_HIGH = 100
    OP_gene, arg_locations, output_locations, constants = winner[:,0:1], winner[:,1:3], winner[:,3:4], winner[:,4:6]

    instruction_idx = np.random.randint(0,OP_gene.shape[0], size=(1) )[0]

    if gene_to_mutate == "gene_setup":
        OP_gene[instruction_idx:instruction_idx+1] = np.random.randint(LOW_SETUP_OPS,UNIQUE_SETUP_OPS, size=(1)).astype(np.float64)
    elif gene_to_mutate == "gene_pred":
        OP_gene[instruction_idx:instruction_idx+1] = np.random.randint(LOW_PRED_OPS,UNIQUE_PRED_OPS, size=(1)).astype(np.float64)
    else:
        OP_gene[instruction_idx:instruction_idx+1] = np.random.randint(LOW_LEARN_OPS,UNIQUE_LEARN_OPS, size=(1)).astype(np.float64)

    arg_locations[instruction_idx] = np.random.randint(0, memory_dict_len,size=arg_locations[instruction_idx].shape).astype(np.float64)
    output_locations[instruction_idx:instruction_idx+1] = np.random.randint(0, memory_dict_len,size=(1)).astype(np.float64)
    constants[instruction_idx:instruction_idx+1] = np.random.uniform(-100,100, size=(1))
    
    return np.hstack((OP_gene, arg_locations, output_locations, constants))

@njit(cache=True)
def _mutate_one_argument(winner, gene_to_mutate, memory_dict_len):

    OP_gene, arg_locations, output_locations, constants = winner[:,0:1], winner[:,1:3], winner[:,3:4], winner[:,4:6]

    instruction_idx = np.random.randint(0,OP_gene.shape[0], size=(1))[0]

    if np.random.random() > 0.5:
        argument_idx = np.random.randint(0,arg_locations[instruction_idx].shape[0], size=(1))[0]
        arg_locations[instruction_idx,argument_idx:argument_idx+1] = np.random.randint(0, memory_dict_len,size=(1,1)).astype(np.float64)
    else:
        output_locations[instruction_idx:instruction_idx+1] = np.random.randint(0, memory_dict_len,size=(1)).astype(np.float64)
        
    #(fitness,(OP_gene, memory_arr, arg_locations, output_locations )
    return np.hstack((OP_gene, arg_locations, output_locations, constants))

@njit(cache=True)
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

    new_metagene[gene_to_mutate] = mutation_function(new_metagene[gene_to_mutate].copy(), gene_to_mutate, memory_dict_len)
    return new_metagene
