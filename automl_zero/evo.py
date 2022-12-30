import time
import numpy as np
from automl_zero.config import *
#FIXME runtime OP lists are in the config file
from automl_zero.ops import *
from automl_zero.memory import initialize_memory_limited
from automl_zero.mutators import mutate_winner
#from automl_zero.utils import apply_output
from numpy.random import randint, sample, choice
from copy import copy
from tqdm import tqdm
from numba import njit
from numba.typed import List, Dict


@njit(cache=True)
def generate_random_gene(NUMBER_OF_OPS = None, 
    MAX_ARG = 2, X_SHAPE = None,y_shape= None ,OP_DEPTH = None,
    CONSTANTS_LOW = -50, CONSTANTS_HIGH= 50,CONSTANTS_MAX =2,
    initialization = "random"):
    
    if initialization == "random":
        OP_gene = np.random.randint(0, NUMBER_OF_OPS , size=(OP_DEPTH,1)).astype(np.float64)
        temp_mem = initialize_memory_limited(X_SHAPE, y_shape) # to have correct arg_list_sizes
        arg_locations = np.random.randint(0, len(temp_mem),size=(OP_DEPTH, MAX_ARG)).astype(np.float64)
        output_locations = np.random.randint(0, len(temp_mem) ,size=(OP_DEPTH,1)).astype(np.float64)
        constants = np.random.uniform(CONSTANTS_LOW,CONSTANTS_HIGH, size=(OP_DEPTH,CONSTANTS_MAX))#gaussian or uniform ??
        #FIXME non int constants

    elif initialization == "zeros":
        OP_gene = np.zeros(shape=(OP_DEPTH,1)).astype(np.float64)
        temp_mem = initialize_memory_limited(X_SHAPE, y_shape) # to have correct arg_list_sizes
        arg_locations = np.zeros(shape=(OP_DEPTH, MAX_ARG)).astype(np.float64)
        output_locations = np.zeros(shape=(OP_DEPTH,1)).astype(np.float64)
        constants = np.zeros(shape=(OP_DEPTH,CONSTANTS_MAX))#gaussian or uniform ??        
        #FIXME non int constants
    
    #OP_gene = np.array(OP_gene, dtype=np.float64)
    #arg_locations = np.array(arg_locations, dtype=np.float64)
    #output_locations = np.array(output_locations, dtype=np.float64)

    return np.hstack((OP_gene, arg_locations, output_locations, constants))

##TODO Classcize this and allow for extentions through inheritance/mixins
@njit(cache=True)
def initialize_population(X = None, y = None, population_count = POPULATION_COUNT,\
                          memory_ref_dict = None, \
                          fitness_func=None, \
                          max_OP_depth = None,
                          setup_OP_dict = None, SETUP_OP_DEPTH = None , \
                          pred_OP_dict = None, PRED_OP_DEPTH = None, \
                          learn_OP_dict = None, LEARN_OP_DEPTH = None, \
                          setup_function = False,
                          learn_function = False,
                          initialization= "zeros"):
    
    
    
    genes_list = List()
    fitness_list = List()
    for _ in range(population_count):
        
        gene_dict = Dict()

        if setup_function:
            # Only one memory for both? 
            gene_setup = generate_random_gene(NUMBER_OF_OPS = SETUP_OP_NUMBER, 
                                              X_SHAPE = X.shape,
                                              y_shape = y_true.shape,
                                              OP_DEPTH = max_OP_depth,
                                              initialization = initialization)
        else:
            gene_setup = None

        gene_pred = generate_random_gene(NUMBER_OF_OPS = PRED_OP_NUMBER,
                                         X_SHAPE = X.shape, 
                                         y_shape = y_true.shape,
                                         OP_DEPTH = max_OP_depth,
                                         initialization = initialization)
        

        if learn_function:
            gene_learn = generate_random_gene(NUMBER_OF_OPS = LEARN_OP_NUMBER, 
                                                X_SHAPE = X.shape,
                                                y_shape = y_true.shape,
                                                OP_DEPTH = max_OP_depth,
                                                initialization = initialization)
            
        else:
            gene_learn = None

        fitness = 0.0
        
        for x_idx in range(len(X)):            
            X_val = X[x_idx:x_idx+1]
            y_val = y[x_idx:x_idx+1]
            #remove previous y used in learn to avoid leakage if repeat value/single example
            memory_ref_dict[2][:] = np.zeros(shape=memory_ref_dict[2].shape )
            #FIXME move out of loop
            #setup function
            if setup_function:
                try:
                    resolve_genome(
                                        gene = gene_setup, \
                                        resolve_depth = SETUP_OP_DEPTH,
                                        memory_ref_dict = memory_ref_dict)
                except:
                    pass #log or some shit

            #predict function
            memory_ref_dict[0][:] = X_val
            try:
                preds = resolve_genome(
                                    gene = gene_pred, \
                                    resolve_depth = PRED_OP_DEPTH,
                                    memory_ref_dict= memory_ref_dict  )# we can supply different OP_dicts to shift meta-levels
                #print(memory_ref_dict)
                
                #fitness is for a single pred here not all? 
                # we do a sum ? 
                fitness += fitness_func(preds, y_true)
            except:
                fitness = 9999999999

            #print("Learn for real")

            if learn_function:
                # Add real y to allow for learning 
                memory_ref_dict[2][:] = y_val
                try:
                    resolve_genome(
                            gene = gene_learn, \
                            resolve_depth = LEARN_OP_DEPTH,
                            memory_ref_dict= memory_ref_dict  )# we can supply different OP_dicts to shift meta-levels
                except:
                    pass

        fitness_list.append(fitness)
        if setup_function:
            gene_dict["gene_setup"] = gene_setup

        gene_dict["gene_pred"] = gene_pred

        if learn_function:
            gene_dict["gene_learn"] = gene_learn

        genes_list.append( 
            gene_dict
             ) 

    return genes_list, fitness_list

@njit(cache=True)
def run_tournament(fitness_list, contestant_indicies):
    min_fitness = 9999999999
    for contestant_idx in contestant_indicies:
        if fitness_list[contestant_idx] <= min_fitness:
            min_fitness = fitness_list[contestant_idx]
            min_idx = contestant_idx
    
    winner_idx = min_idx
    
    return winner_idx

@njit(cache=True)
def get_best_fitness(fitness_list):
    min_fitness = 9999999999
    for gene_idx in range(len(fitness_list)):
        if min_fitness >= fitness_list[gene_idx]:
            min_fitness = fitness_list[gene_idx]
            min_idx = gene_idx
    return min_idx

## TODO this can be vectorized
@njit(cache=True)
def run_evolution(X, y, iters = 100000,
                fitness_func = None, \
                population_list = None, 
                fitness_list = None,
                setup_function = True,
                setup_OP_dict = setup_OPs, 
                SETUP_OP_DEPTH = None,
                pred_OP_dict = pred_OPs,
                PRED_OP_DEPTH = None,
                learn_function = True,
                learn_OP_dict = learn_OPs,
                LEARN_OP_DEPTH = None,
                N=10,
                memory_ref_dict = None):
    """
    There are 2 modes of referencing 
    """
    #print("Running Evolution")
    #memory_ref_dict = initialize_memory_limited(X_shape = X.shape,y_shape = y.shape, scalars=5, vectors=5, matricies=5)
    #start_time = time.time()
    for i in range(iters):
        
        if i%10000 == 0:
            best_fitness_idx = get_best_fitness(fitness_list)
            print("BEST FITNESS TO DATE: ")
            print(fitness_list[best_fitness_idx])
            if fitness_list[best_fitness_idx] == 0:
                print("ITER Escaped at: ")
                print(i)
                return population_list,fitness_list, i
        
        contestant_indicies = choice(np.arange(len(fitness_list)), size=N, replace=False)
        tournament_winner_idx = run_tournament(fitness_list, contestant_indicies)        
        new_metagene = mutate_winner(population_list[tournament_winner_idx], len(memory_ref_dict))
        ## FIXME How can this cause issues? Leak zeros inappropriatly?
        new_fitness = 0

        memory_ref_dict[2][:] = np.zeros(shape=memory_ref_dict[2].shape )
        #setup function
        #print("Setting Up")
        if setup_function:
            resolve_genome(
                                gene = new_metagene["gene_setup"], \
                                memory_ref_dict = memory_ref_dict,
                                resolve_depth = SETUP_OP_DEPTH)

        #TODO add in f(x)? profile
        #TODO profile this zip vs range(len())
        for X_val, y_val in zip(X,y):        
            
            #remove previous y used in learn to avoid leakage if repeat value/single example
            #print("Here?")
            memory_ref_dict[2][:] = np.zeros(shape=memory_ref_dict[2].shape )
            #print(memory_ref_dict.keys())
            #print("Predicting")            
            #predict function
            memory_ref_dict[0][:] = X_val
            preds = resolve_genome(
                                gene = new_metagene["gene_pred"], \
                                memory_ref_dict= memory_ref_dict,
                                resolve_depth=PRED_OP_DEPTH  )# we can supply different OP_dicts to shift meta-levels
            
            #fitness is for a single pred here not all? 
            # we do a sum ? 
            new_fitness += fitness_func(preds, y_val)
            #print("Learning")
            if learn_function:
                # Add real y to allow for learning 
                memory_ref_dict[2][:] = y_val
                #print("is it this?")

                resolve_genome(
                        gene = new_metagene["gene_learn"], \
                        memory_ref_dict= memory_ref_dict,
                        resolve_depth=LEARN_OP_DEPTH  )# we can supply different OP_dicts to shift meta-levels
                #print("is it this?")

            #print("One Done ")
        ## Append to population list
        population_list.append( new_metagene )
        population_list.pop(0)
        fitness_list.append(new_fitness)
        fitness_list.pop(0)
    #end_time = time.time()
    #seconds_spent = end_time-start_time
    #iters_per_second = iters/seconds_spent
    #print(f"ITERS PER SECOND: {iters_per_second}" )
    return population_list, fitness_list, i


#FIXME set OP_dict for max_OP_depth and then control exit condition from loop
# to allow for numba to have fixed type arguments
@njit(cache=True)
def resolve_genome(
    gene = None,
    resolve_depth = PRED_OP_NUMBER,
    memory_ref_dict = None,
    constants_flag = False
):
    #print("Resolving Genome")
    genome_OPs, args_locations, output_locations, constants = gene[:,0], gene[:,1:3], gene[:,3], gene[:,4:6]

    for gene_idx in range(0,resolve_depth):
        #print("Evaluating")
        op_number    = genome_OPs[gene_idx]
        #print(op_number)

        if constants_flag:
            #con_args_0 = constants[gene_idx][0]
            #con_args_1 = constants[gene_idx][1]
            #result = OP_to_apply(con_args_0, con_args_1)
            pass

        else:
            #print(op_args_0)
            #print(args_locations[gene_idx][0])
            #print(memory_ref_dict[args_locations[gene_idx][0]])
            op_args_0 = memory_ref_dict[int(args_locations[gene_idx][0])]
            op_args_1 = memory_ref_dict[int(args_locations[gene_idx][1])]
            #op_args_1 = np.resize(op_args_1, op_args_0.shape)
           # #op_args_1 = memory_ref_dict[int(args_locations[gene_idx][1])]
            #result = OP_to_apply(op_args_0, op_args_1)
            #FIXME add consts to the other condition
            if op_number == 0:
                continue
            elif op_number == 1:
                result = add_scalar(op_args_0, op_args_1)
            elif op_number == 2:
                result = add_scalar(op_args_0, -op_args_1)
            elif op_number == 3:
                result = multiply_scalar(op_args_0, op_args_1)
            elif op_number == 4:
                result = divide_scalar(op_args_0, op_args_1)
            elif op_number == 5:
                result = abs_scalar(op_args_0, op_args_1)
            elif op_number == 6:
                result = reciprocal_scalar(op_args_0 )
            elif op_number == 7:
                result = sin_scalar(op_args_0)
            elif op_number == 8:
                result = cos_scalar(op_args_0)
            elif op_number == 9:
                result = tan_scalar(op_args_0)
            elif op_number == 10:
                result = arcsin_scalar(op_args_0)
            elif op_number == 11:
                result = arccos_scalar(op_args_0)
            elif op_number == 12:
                result = arctan_scalar(op_args_0)
        
        output_idx = int(output_locations[gene_idx])
        #output_arr = memory_ref_dict[output_idx]
        
        try:
            if result is not None: 
                memory_ref_dict[output_idx][:] = result
            else:
                memory_ref_dict[output_idx][:] = np.zeros(shape= memory_ref_dict[output_idx].shape)
        except:
            #memory_ref_dict[output_idx][:] = np.resize(result, output_arr.shape)
            memory_ref_dict[output_idx][:] = np.zeros(shape= memory_ref_dict[output_idx].shape)

    #print("Genome Resolved")
    return memory_ref_dict[1]
