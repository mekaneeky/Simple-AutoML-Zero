import numpy as np 

from random import choice 
from copy import deepcopy
from automl_zero.ops import OP_dict_sizes
from automl_zero.memory import initialize_memory_limited

## TODO use globals ?
def _mutate_all(winner,gene_to_mutate, memory_dict_len
,CONSTANTS_LOW = -100,CONSTANTS_HIGH = 100):

    OP_gene, arg_locations, output_locations, constants = winner
    
    OP_gene = np.random.randint(0,OP_dict_sizes[gene_to_mutate], size=OP_gene.shape)
    arg_locations = np.random.randint(0, memory_dict_len,size=(arg_locations.shape)).astype(int)
    output_locations = np.random.randint(0, memory_dict_len ,size=(output_locations.shape)).astype(int)
    constants = np.random.uniform(CONSTANTS_LOW,CONSTANTS_HIGH, size=constants.shape)

    return OP_gene, arg_locations, output_locations,constants

def _mutate_add_or_remove_one_instruction(winner, gene_to_mutate,  memory_dict_len
,CONSTANTS_LOW = -100,CONSTANTS_HIGH = 100):

    OP_gene, arg_locations, output_locations, constants = winner

    instruction_idx = np.random.randint(0,OP_gene.shape[0], size=(1) )[0]

    OP_gene[instruction_idx] = np.random.randint(0,OP_dict_sizes[gene_to_mutate], size=OP_gene[instruction_idx].shape)
    arg_locations[instruction_idx] = np.random.randint(0, memory_dict_len,size=arg_locations[instruction_idx].shape).astype(int)
    output_locations[instruction_idx] = np.random.randint(0, memory_dict_len,size=output_locations[instruction_idx].shape).astype(int)
    constants[instruction_idx] = np.random.uniform(CONSTANTS_LOW,CONSTANTS_HIGH, size=arg_locations[instruction_idx].shape)
    
    return OP_gene, arg_locations, output_locations, constants

def _mutate_one_argument(winner, gene_to_mutate, memory_dict_len):

    OP_gene, arg_locations, output_locations, constants = winner

    instruction_idx = np.random.randint(0,OP_gene.shape[0], size=(1))[0]

    if np.random.random() > 0.5:
        argument_idx = np.random.randint(0,arg_locations[instruction_idx].shape[0], size=(1))[0]
        arg_locations[instruction_idx,argument_idx] = np.random.randint(0, memory_dict_len,size=(1)).astype(int)
    else:
        output_locations[instruction_idx] = np.random.randint(0, memory_dict_len,size=output_locations[instruction_idx].shape).astype(int)
        
    #(fitness,(OP_gene, memory_arr, arg_locations, output_locations )
    return OP_gene, arg_locations, output_locations, constants


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

    if winner_metagene["gene_setup"] != None and winner_metagene["gene_learn"]:
        gene_to_mutate = choice(["gene_setup", "gene_pred", "gene_learn"])
    elif winner_metagene["gene_setup"] != None:
        gene_to_mutate = choice(["gene_setup", "gene_pred"])
    else:
        gene_to_mutate = "gene_pred"

    mutations = [_mutate_one_argument, _mutate_all,  _mutate_add_or_remove_one_instruction ]
    mutation_function = choice(mutations)
    new_metagene = deepcopy(winner_metagene) # Should we have a state of all functions 
                                             # in a dictionary where 42:function_for_all()
    new_metagene[gene_to_mutate] = mutation_function(new_metagene[gene_to_mutate], gene_to_mutate, memory_dict_len)
    new_metagene["fitness"] = None
    return new_metagene
