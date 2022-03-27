import numpy as np
from automl_zero.config import *
from automl_zero.ops import OP_dict_basic
from random import choice, randint, sample
from copy import deepcopy
from tqdm import tqdm
#from numba import jit, njit
## 

def initialialize_population(X_arr, y_true= None, fitness_func= None, \
                     population_count = POPULATION_COUNT, op_depth = None, \
                     memory_size = MEMORY_SIZE , OP_dict = OP_dict_basic, \
                     max_arg = 2, element_type = "matrix"):
    genes_list = []
    #FIXME make memory arr a global variable
    memory_arr = initialize_memory(memory_size = MEMORY_SIZE, element_type = element_type, matrix_size= MATRIX_SIZE, start=-10, end=10)
    
    for _ in range(population_count):    
        
        OP_gene = np.random.randint(0,len(OP_dict_basic), size=(OP_DEPTH,1))
        arg_locations = np.random.randint(0, len(memory_arr),size=(OP_DEPTH, max_arg)).astype(int)
        output_locations = np.random.randint(0, len(memory_arr) ,size=(OP_DEPTH, 1)).astype(int)    
        preds = resolve_genome(X_arr,OP_gene, deepcopy(memory_arr), arg_locations, output_locations, OP_dict  )
        fitness = fitness_func(preds, y_true)
        genes_list.append( (fitness,(OP_gene, deepcopy(memory_arr), arg_locations, output_locations )  ) )

    return genes_list


def initialize_memory(memory_size = MEMORY_SIZE, element_type = "scalar", matrix_size= MATRIX_SIZE, start=-100, end=100):
    """
        weight_type: int or float, specifies whether to 
    """    

    if element_type == "scalar":
        scalar_memory = np.random.uniform(start,end, size=(memory_size,))
        return scalar_memory
    elif element_type == "vector":
        vector_memory = np.random.normal(loc=0 , scale=1, size=(memory_size,matrix_size[0]) )
        return vector_memory
    elif element_type == "matrix":
        matrix_memory = np.random.normal(loc=0 , scale=1, size=(memory_size,matrix_size[0],matrix_size[1]) )
        return matrix_memory

def _mutate_all(winner):
    _, (OP_gene,memory_arr, arg_locations, output_locations) = winner

    #scalar_memory_arr, vector_memory_arr, matrix_memory_arr = initialize_memory(memory_size = MEMORY_SIZE, element_type = "scalar", matrix_size= (2,2), start=-100, end=100)
    OP_gene = np.random.randint(0,len(OP_dict_basic), size=(OP_DEPTH,1))
    arg_locations = np.random.randint(0, len(memory_arr),size=(arg_locations.shape)).astype(int)
    output_locations = np.random.randint(0, len(memory_arr) ,size=(output_locations.shape)).astype(int)

    return (-9999, (OP_gene,memory_arr, arg_locations, output_locations))

def _mutate_add_or_remove_one_instruction(winner):
    _, (OP_gene, memory_arr, arg_locations, output_locations) = deepcopy(winner)

    instruction_idx = np.random.randint(0,OP_DEPTH, size=(1) )[0]

    OP_gene[instruction_idx] = np.random.randint(0,len(OP_dict_basic), size=OP_gene[instruction_idx].shape)
    arg_locations[instruction_idx] = np.random.randint(0, len(memory_arr),size=arg_locations[instruction_idx].shape).astype(int)
    output_locations[instruction_idx] = np.random.randint(0, len(memory_arr),size=output_locations[instruction_idx].shape).astype(int)

    return (-9999, (OP_gene,memory_arr, arg_locations, output_locations))

def _mutate_one_argument(winner):
    _, (OP_gene, memory_arr,arg_locations, output_locations) = deepcopy(winner)

    instruction_idx = np.random.randint(0,OP_DEPTH, size=(1))[0]

    if np.random.random() > 0.5:
        argument_idx = np.random.randint(0,arg_locations[instruction_idx].shape[0], size=(1))[0]
        arg_locations[instruction_idx,argument_idx] = np.random.randint(0, len(memory_arr),size=(1)).astype(int)
    else:
        output_locations[instruction_idx] = np.random.randint(0, len(memory_arr),size=output_locations[instruction_idx].shape).astype(int)

    return (-9999, (OP_gene,memory_arr, arg_locations, output_locations))    

def mutate_winner(winner):

    mutations = [_mutate_one_argument, _mutate_all,  _mutate_add_or_remove_one_instruction ]
    mutation_function = choice(mutations)
    return mutation_function(winner)

def run_tournament(contestants):
    min_fitness = float("inf")
    for idx in range(len(contestants)):
        if contestants[idx][0] < min_fitness:
            min_fitness = contestants[idx][0]
            min_idx = idx
    tournament_winner = contestants[min_idx]
    return tournament_winner


## TODO this can be vectorized
def run_evolution(X_arr, true_y, iters = 100000,fitness_func = None, reward_to_gene_list = None, N=10, population = POPULATION_COUNT):
    
    for i in tqdm(range(iters)):   
        
        min_fitness = float("inf")
        if i%1000 == 0:
            for fitness, (_,_,_,_) in reward_to_gene_list:
                if min_fitness > fitness:
                    min_fitness = fitness
            print("BEST FITNESS TO DATE: {}".format(min_fitness))

        #TODO use arrays not lists they are faster
        contestants = sample(reward_to_gene_list, N)
        
        tournament_winner = run_tournament(contestants)
        import pdb;pdb.set_trace()
        fitness, (OP_gene,memory_arr, args_gene, out_gene) = tournament_winner
        new_child = mutate_winner(tournament_winner)
        preds = resolve_genome(X_arr,new_child[1][0], new_child[1][1], new_child[1][2], new_child[1][3], OP_dict_basic  )
        fitness = fitness_func(preds, true_y)
        if np.isnan(fitness):
            fitness = MIN_FITNESS

        reward_to_gene_list.append(( fitness, (new_child[1][0], new_child[1][1], new_child[1][2], new_child[1][3])) )
        reward_to_gene_list.pop(0)        
        
    return reward_to_gene_list

## Here is where each OP needs to be applied based on the OP dimension in 
## (OP#:(OP, MAX_ARGS, INPUT_TYPE (0:Scalar,1:Vector,2:Matrix,3:SC/Vector,4:SC/Matrix),
#      OUTPUT_TYPE (0:Scalar,1:Vector,2:Matrix)))
## If special accomodations are needed now what?
## Will attempt to force ndim == 3 for all
def resolve_genome(X_vals,genome_OPs = None, memory_arr = None, \
                   args_locations= None, output_locations = None , \
                   OP_dict = OP_dict_basic):
    final_results_arr = np.empty_like(X_vals).astype(float)
    #temp_memory = deepcopy(memory_arr)

    assert X_vals.ndim == 3 ## batch, x,y. Can easily be expanded to n dims 
    for X_idx in range(X_vals.shape[0]):
        memory_arr[INPUT_ADDR][0:X_vals.shape[1],0:X_vals.shape[2]] = X_vals[X_idx]
       
        for gene_idx in range(0,len(genome_OPs)):

            ## Instead of being listed in the genome, they can be different OPs for different cases            
            ## FIXME remove input + output type? 
            OP_to_apply, max_args, _, _ = OP_dict[genome_OPs[gene_idx][0]]

            ## We get max_args locations in memory + output location in memory
            args = memory_arr [ args_locations[gene_idx,:max_args]]
            results = OP_to_apply(*args)
            memory_arr[output_locations[gene_idx][0]] [0:results.shape[0],0:results.shape[1]] = results
            if np.isnan(memory_arr[output_locations[gene_idx]]).any():
                memory_arr[output_locations[gene_idx]] = MIN_VAL
        #try:
        final_results_arr[X_idx] = memory_arr[OUTPUT_ADDR].reshape(final_results_arr[X_idx].shape)
        
    return final_results_arr

def mse_fitness(preds, HALVED):
    return np.sum(np.abs(preds - HALVED))
