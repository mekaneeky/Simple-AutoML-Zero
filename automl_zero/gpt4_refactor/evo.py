from automl_zero.hierarchical.config import *
from automl_zero.memory import initialize_memory_limited
from automl_zero.ops import *
from automl_zero.gpt4_refactor.mutators import mutate_winner, mutate_combination_winner
from automl_zero.hierarchical.op_combination import create_OP_population, get_ops_from_meta_level, _generate_cached_metalevels
from numpy.random import choice
from numba import njit
from numba.typed import List, Dict

#@njit(cache=True)
def base_resolve_op(op_number):
    op_number = int(op_number)
    if op_number == 0:
        result = identity#FIXME
    elif op_number == 1:
        result = add_scalar
    elif op_number == 2:
        result = add_scalar
    elif op_number == 3:
        result = multiply_scalar
    elif op_number == 4:
        result = divide_scalar
    elif op_number == 5:
        result = abs_scalar
    elif op_number == 6:
        result = reciprocal_scalar
    elif op_number == 7:
        result = sin_scalar
    elif op_number == 8:
        result = cos_scalar
    elif op_number == 9:
        result = tan_scalar
    elif op_number == 10:
        result = arcsin_scalar
    elif op_number == 11:
        result = arccos_scalar
    elif op_number == 12:
        result = arctan_scalar

    return result


def generate_random_values(low, high, shape, dtype):
    return np.random.uniform(low, high, size=shape).astype(dtype)

def initialize_values(value, shape, dtype):
    return np.full(shape, value, dtype=dtype)

def initialize_memory(X_SHAPE, y_shape, scalars,vectors, matricies):
    # Add your memory initialization logic here
    return initialize_memory_limited(X_shape = X_SHAPE , y_shape = y_shape, scalars = scalars, vectors = vectors, matricies = matricies)

#Avoiding base ops due to evil bug FIXME
# Will let evolution deal with it through evolving metalevel 1 genomes if needed
def generate_random_hierarchical_gene(NUMBER_OF_METALEVEL_OPS, MAX_ARG, X_SHAPE, y_shape, OP_DEPTH, CONSTANTS_LOW, CONSTANTS_HIGH, CONSTANTS_MAX, initialization, scalars,
                                             vectors, matricies,METALEVEL_COUNT):
    init_func = generate_random_values if initialization == "random" else initialize_values
    base_OP_idx = init_func(0, NUMBER_OF_METALEVEL_OPS , (OP_DEPTH,1), np.int8)
    OP_metalevel = init_func(0, METALEVEL_COUNT , (OP_DEPTH,1), np.int8)

    temp_mem = initialize_memory(X_SHAPE, y_shape,scalars,vectors, matricies,)
    arg_locations = init_func(0, len(temp_mem),(OP_DEPTH, MAX_ARG), np.float64)
    output_locations = init_func(0, len(temp_mem) ,(OP_DEPTH,1), np.float64)
    constants = init_func(CONSTANTS_LOW, CONSTANTS_HIGH, (OP_DEPTH,CONSTANTS_MAX), np.float64)

    return np.hstack((base_OP_idx, OP_metalevel, arg_locations, output_locations, constants))


def generate_gene(X_shape, y_shape, condition, NUMBER_OF_METALEVEL_OPS):
    return generate_random_hierarchical_gene(NUMBER_OF_METALEVEL_OPS=NUMBER_OF_METALEVEL_OPS, 
                                             MAX_ARG=2, 
                                             X_SHAPE=X_shape, 
                                             y_shape=y_shape, 
                                             OP_DEPTH=max_OP_depth, 
                                             CONSTANTS_LOW=-100, 
                                             CONSTANTS_HIGH=100, 
                                             CONSTANTS_MAX=1, 
                                             initialization="random" if random_initialization else "zeros", 
                                             scalars=scalars,
                                             vectors=vectors, 
                                             matricies=matricies, 
                                             METALEVEL_COUNT=METALEVEL_COUNT) if condition else None




#@njit(cache=True)
def initialize_gene_population(X, y, memory_ref_dict, fitness_func, cached_metalevels):
    genes_list = List()
    fitness_list = List()
    NUMBER_OF_METALEVELS, NUMBER_OF_META_OPS, FIRST_LEVEL_TOTAL_OPS = cached_metalevels.shape

    for i in range(POPULATION_COUNT):
        print(i) if i % 10 == 0 else None

        gene_setup, gene_pred, gene_learn = initialize_genes(X, y, memory_ref_dict, NUMBER_OF_META_OPS)

        fitness, accuracy = evaluate_genes(X, y, memory_ref_dict, gene_setup, gene_pred, gene_learn, fitness_func, cached_metalevels)

        fitness_list.append(-accuracy / len(X)) if accuracy else fitness_list.append(fitness)
        genes_dict = Dict()
        genes_dict["gene_setup"] = gene_setup 
        genes_dict["gene_pred"] = gene_pred
        genes_dict["gene_learn"] = gene_learn
        genes_list.append(genes_dict)

    return genes_list, fitness_list

def initialize_OP_gene_population(X, y, memory_ref_dict, fitness_func, reference_gene,resolve_depth,
                    NUMBER_OF_METALEVELS, NUMBER_OF_META_OPS, FIRST_LEVEL_TOTAL_OPS):
    OP_genes_list = List()
    OP_cache_list = List()
    fitness_list = List()
    
    
    
    for i in range(POPULATION_COUNT):
        print(i) if i % 10 == 0 else None

        OP_gene, OP_cached_gene = create_OP_gene(METALEVEL_COUNT, NUMBER_OF_META_OPS, PRIOR_LEVEL_OPS)
        if type(reference_gene) == list:
            fitness = 0
            accuracy = 0
            for metagene in reference_gene:
                _fitness, _accuracy = evaluate_individual(X, y, OP_cached_gene, metagene, fitness_func, memory_ref_dict,resolve_depth, MAX_ARG)
                fitness += _fitness
                accuracy += _accuracy #FIXME potential bug
        else:
            fitness, accuracy = evaluate_individual(X, y, OP_cached_gene, reference_gene, fitness_func, memory_ref_dict,resolve_depth, MAX_ARG)

        if type(reference_gene) == list:
            fitness_list.append(-accuracy / len(X) / len(reference_gene)) if accuracy else fitness_list.append(fitness)
        else:
            fitness_list.append(-accuracy / len(X)) if accuracy else fitness_list.append(fitness)
        OP_genes_list.append(OP_gene)
        OP_cache_list.append(OP_cached_gene)
        

    return OP_genes_list, OP_cache_list, fitness_list




def initialize_genes(X, y, memory_ref_dict,NUMBER_OF_META_OPS):
    gene_setup = generate_gene(X[0].shape, y[0].shape, SETUP_FUNCTION, NUMBER_OF_META_OPS)
    gene_pred = generate_gene(X[0].shape, y[0].shape, True, NUMBER_OF_META_OPS)
    gene_learn = generate_gene(X[0].shape, y[0].shape, LEARN_FUNCTION, NUMBER_OF_META_OPS)
    return gene_setup, gene_pred, gene_learn


def evaluate_genes(X, y, memory_ref_dict, gene_setup, gene_pred, gene_learn, fitness_func, cached_metalevels):
    fitness = 0.0
    accuracy = 0.0
    for x_idx in range(len(X)):
        X_val = X[x_idx:x_idx+1]
        y_val = y[x_idx:x_idx+1]

        memory_ref_dict[2][:] = np.zeros(shape=memory_ref_dict[2].shape)  # Remove previous y used in learn to avoid leakage

        if setup_function:
            hierarchical_resolve_genome(gene=gene_setup, cached_metalevels=cached_metalevels, resolve_depth=SETUP_OP_DEPTH,
                                        memory_ref_dict=memory_ref_dict, MAX_ARG=MAX_ARG)

        memory_ref_dict[0][:] = X_val
        try:
            preds = hierarchical_resolve_genome(gene=gene_pred, cached_metalevels=cached_metalevels["gene_pred"], resolve_depth=PRED_OP_DEPTH,
                                                memory_ref_dict=memory_ref_dict, MAX_ARG=MAX_ARG)
            fitness += fitness_func(preds, y_val)
            accuracy += np.sum(np.floor(preds) == y_val)
        except:
            accuracy = 0
            fitness = float(9999999999.000)

        if learn_function:
            memory_ref_dict[2][:] = y_val
            try:
                hierarchical_resolve_genome(gene=gene_learn, cached_metalevels=cached_metalevels["gene_learn"],
                                            resolve_depth=LEARN_OP_DEPTH, memory_ref_dict=memory_ref_dict, MAX_ARG=MAX_ARG)
            except:
                pass

    return fitness, accuracy

def retrieve_op(op_value, metalevel, cached_metalevels):
    #return [op_value] if metalevel == 0 else cached_metalevels[int(metalevel), int(op_value)]
    return cached_metalevels[int(metalevel), int(op_value)]

def execute_op(op, first_OP_flag, temp_result, op_arg_0, op_arg_1):
    if first_OP_flag:
        return op(op_arg_0, op_arg_1), False
    else:
        return op(temp_result, op_arg_1), first_OP_flag

def update_memory(temp_result, memory_ref_dict, output_idx):
    try:
        if temp_result is not None: 
            memory_ref_dict[output_idx][:] = temp_result
        else:
            memory_ref_dict[output_idx][:] = np.zeros(shape= memory_ref_dict[output_idx].shape)

    except:
        #memory_ref_dict[output_idx][:] = np.resize(result, output_arr.shape)
        memory_ref_dict[output_idx][:] = np.zeros(shape= memory_ref_dict[output_idx].shape)

    return memory_ref_dict

def hierarchical_resolve_genome(gene, cached_metalevels, memory_ref_dict, resolve_depth = None, constants_flag = False, OP_depth = None, MAX_ARG = None):
    genome_OPs, metalevel_gene, args_locations, output_locations, constants = gene[:,0:1], gene[:,1:2], gene[:,2:2+MAX_ARG], gene[:,2+MAX_ARG:3+MAX_ARG],gene[:,3+MAX_ARG:]
    
    for gene_idx in range(len(genome_OPs)):
        OP_value = genome_OPs[gene_idx].astype(np.int8)
        metalevel = metalevel_gene[gene_idx].astype(np.int8)

        first_level_OPs = retrieve_op(OP_value, metalevel, cached_metalevels)

        op_args_0 = memory_ref_dict[int(args_locations[gene_idx][0])]
        op_args_1 = memory_ref_dict[int(args_locations[gene_idx][1])]
        first_OP_flag = True

        for OP_idx in first_level_OPs:
            if int(OP_idx) == -1:
                break
            OP_to_execute = base_resolve_op(OP_idx)
            #if OP_to_execute is None:
            #    break
            try:
                temp_result, first_OP_flag = execute_op(OP_to_execute, first_OP_flag, temp_result, op_args_0, op_args_1)
            except UnboundLocalError:
                temp_result, first_OP_flag = execute_op(OP_to_execute, first_OP_flag, op_args_0, op_args_0, op_args_1)

        output_idx = int(output_locations[gene_idx])
        memory_ref_dict = update_memory(temp_result, memory_ref_dict, output_idx)

    return memory_ref_dict[1]

#idx = (-arr).argsort()[:n]
def run_tournament(fitness_list, contestant_indices):
    """
    Run a tournament among selected indices in the population. 
    Return the index of the individual with the highest fitness.
    """
    # Array of the fitness values of the contestants
    contestant_fitnesses = [fitness_list[i] for i in contestant_indices]

    # Get the index of the individual with the highest fitness among the contestants
    winner_index = np.argmax(contestant_fitnesses)

    # Return the index of the winning individual in the original population
    return contestant_indices[winner_index]

def tournament_selection(fitness_list, N):
    contestant_indicies = choice(np.arange(len(fitness_list)), size=N, replace=False)
    return run_tournament(fitness_list, contestant_indicies) 

def evaluate_individual(X, y, cached_metalevels, new_metagene, fitness_func, memory_ref_dict, resolve_depth, MAX_ARG):
    new_fitness = 0
    new_accuracy = 0

    if setup_function:
            hierarchical_resolve_genome(
                                gene=new_metagene["gene_setup"], 
                                cached_metalevels= cached_metalevels,
                                memory_ref_dict=memory_ref_dict,
                                resolve_depth=SETUP_OP_DEPTH,
                                MAX_ARG=MAX_ARG)

    for X_val, y_val in zip(X,y):        
        # Clear previous values to avoid leakage
        memory_ref_dict[2][:] = np.zeros(shape=memory_ref_dict[2].shape )

        # Generate predictions
        memory_ref_dict[0][:] = X_val
        preds = hierarchical_resolve_genome(
                            gene=new_metagene["gene_pred"], 
                            cached_metalevels= cached_metalevels,

                            memory_ref_dict=memory_ref_dict,
                            resolve_depth=resolve_depth,
                            MAX_ARG=MAX_ARG)

        new_fitness += fitness_func(preds, y_val)
        new_accuracy += np.sum(np.floor(preds) == y_val)
    
        if learn_function:
            # Add real y to allow for learning 
            memory_ref_dict[2][:] = y_val

            hierarchical_resolve_genome(
                    gene=new_metagene["gene_learn"], 
                    cached_metalevels= cached_metalevels,
                    memory_ref_dict=memory_ref_dict,
                    resolve_depth=LEARN_OP_DEPTH,
                    MAX_ARG=MAX_ARG)

    return new_fitness, new_accuracy

def hierarchical_run_gene_evolution(X, y, iters=100000, evaluate_steps=10000, fitness_func=None, population_list=None, 
                                    fitness_list=None, setup_function=True, setup_OP_dict=setup_OPs, SETUP_OP_DEPTH=None, 
                                    pred_OP_dict=pred_OPs, PRED_OP_DEPTH=None, learn_function=True, learn_OP_dict=learn_OPs, 
                                    LEARN_OP_DEPTH=None, N=10, memory_ref_dict=None, accuracy=False, cached_metalevels=None,
                                    MAX_ARG=2):
    """
    There are 2 modes of referencing 
    """
    for i in range(iters):
        if i % 10 == 0:
            print(f"BEST FITNESS at iteration {i} is {max(fitness_list)}")
        
        ## Tournament selection
        tournament_winner_idx = tournament_selection(fitness_list, N)
        new_metagene = mutate_winner(population_list[tournament_winner_idx], len(memory_ref_dict))
        memory_ref_dict[2][:] = np.zeros(shape=memory_ref_dict[2].shape )
        
        new_fitness, new_accuracy = evaluate_individual(X, y, cached_metalevels, new_metagene, fitness_func, memory_ref_dict, PRED_OP_DEPTH, MAX_ARG)
        
        ## Append to population list
        population_list.append(new_metagene) #FIXME where to add the cached_metalevels
        population_list.pop(0)
        fitness_list.append(-new_accuracy/len(X) if accuracy else new_fitness)
        fitness_list.pop(0)

    return population_list, fitness_list, i



def hierarchical_run_op_evolution(OP_population, OP_cache,OP_fitness_list, X, y, iters=1000, evaluate_steps=10000,reference_metagene= None, fitness_func=None, 
                                   N=10, memory_ref_dict=None, accuracy=False,resolve_depth=None, MAX_ARG=2):
    """
    This function is focused on evolving the hierarchical operations (OP genes) 
    rather than the function genes.
    """
    ## This function is not yet complete. This is a GPT4 generated blueprint
    for i in range(iters):
        if i % 10 == 0:
            print(f"BEST FITNESS at iteration {i} is {max(OP_fitness_list)}")
        
        ## Tournament selection
        tournament_winner_idx = tournament_selection(OP_fitness_list, N)
        new_op_gene, new_cached_gene = mutate_combination_winner(OP_population[tournament_winner_idx])
        
        ## Evaluate the new OP gene 
        if type(reference_metagene) == list:
            new_fitness = 0
            new_accuracy = 0
            for single_reference_metagene in reference_metagene:
                _new_fitness, _new_accuracy = evaluate_individual(X, y, new_cached_gene, \
                                                            single_reference_metagene, fitness_func, \
                                                            memory_ref_dict,resolve_depth, MAX_ARG)
                new_fitness += _new_fitness
                new_accuracy += _new_accuracy
        else:
            new_fitness, new_accuracy = evaluate_individual(X, y, new_cached_gene, \
                                                            reference_metagene, fitness_func, \
                                                            memory_ref_dict,resolve_depth, MAX_ARG)
        ## Append to OP population list
        OP_population.append(new_op_gene)
        OP_population.pop(0)
        OP_cache.append(new_cached_gene)
        OP_cache.pop(0)
        OP_fitness_list.append(-new_accuracy/len(X) if accuracy else new_fitness)
        OP_fitness_list.pop(0)

    return OP_population, OP_cache, OP_fitness_list, i


def cyclical_evolution(X, y, iters=100000, evaluate_steps=10000, fitness_func=None, 
                       gene_population_list=None, gene_fitness_list=None, 
                       op_population=None, op_cache=None, op_fitness_list=None,
                       setup_function=True, setup_OP_dict=setup_OPs, SETUP_OP_DEPTH=None, 
                       pred_OP_dict=pred_OPs, PRED_OP_DEPTH=None, learn_function=True, 
                       learn_OP_dict=learn_OPs, LEARN_OP_DEPTH=None, N=10, 
                       memory_ref_dict=None, accuracy=False, MAX_ARG=2, cycles=10):
    
    for cycle in range(cycles):
        print(f"Starting Cycle {cycle + 1}")
        
        # Select the winner of the function genes as reference_metagene for OP genes evolution
        winner_idx_function = tournament_selection(gene_fitness_list, N)
        reference_metagene = gene_population_list[winner_idx_function]
        
        # Evolve the OP genes using the winner of the function genes as reference_metagene
        evolved_op_population, evolved_op_cache, evolved_op_fitness_list, _ = hierarchical_run_op_evolution(
            OP_population=op_population, 
            OP_cache=op_cache, 
            X=X, 
            y=y, 
            iters=iters, 
            evaluate_steps=evaluate_steps, 
            fitness_func=fitness_func, 
            fitness_list=op_fitness_list, 
            N=N, 
            memory_ref_dict=memory_ref_dict, 
            accuracy=accuracy, 
            MAX_ARG=MAX_ARG,
            reference_metagene=reference_metagene  # Use the winner of the function genes as reference_metagene
        )
        
        # Select the winner of the OP genes as cached_metalevels for function genes evolution
        winner_idx_op = tournament_selection(evolved_op_fitness_list, N)
        cached_metalevels = evolved_op_cache[winner_idx_op]
        
        # Evolve the function genes using the winner of the OP genes as cached_metalevels
        evolved_gene_population, evolved_gene_fitness_list, _ = hierarchical_run_gene_evolution(
            X=X, 
            y=y, 
            iters=iters, 
            evaluate_steps=evaluate_steps, 
            fitness_func=fitness_func, 
            population_list=gene_population_list, 
            fitness_list=gene_fitness_list, 
            setup_function=setup_function, 
            setup_OP_dict=setup_OP_dict, 
            SETUP_OP_DEPTH=SETUP_OP_DEPTH, 
            pred_OP_dict=pred_OP_dict, 
            PRED_OP_DEPTH=PRED_OP_DEPTH, 
            learn_function=learn_function, 
            learn_OP_dict=learn_OP_dict, 
            LEARN_OP_DEPTH=LEARN_OP_DEPTH, 
            N=N, 
            memory_ref_dict=memory_ref_dict, 
            accuracy=accuracy, 
            cached_metalevels=cached_metalevels,  # Use the winner of the OP genes as cached_metalevels
            MAX_ARG=MAX_ARG
        )
        
        # Update the populations and fitness lists for the next cycle
        op_population = evolved_op_population
        op_cache = evolved_op_cache
        op_fitness_list = evolved_op_fitness_list
        gene_population_list = evolved_gene_population
        gene_fitness_list = evolved_gene_fitness_list
        
    return evolved_gene_population, evolved_gene_fitness_list, evolved_op_population, evolved_op_cache, evolved_op_fitness_list

def select_functional_genes_for_op_evaluation(gene_population_list, gene_fitness_list, N, count):
    selected_functional_genes = []
    for _ in range(count):
        winner_idx = tournament_selection(gene_fitness_list, N)
        selected_functional_genes.append(gene_population_list[winner_idx])
    return selected_functional_genes

def select_op_genes_for_functional_evaluation(op_population, op_cache, op_fitness_list, N, count):
    selected_op_genes = []
    selected_op_cache = []
    for _ in range(count):
        winner_idx = tournament_selection(op_fitness_list, N)
        selected_op_genes.append(op_population[winner_idx])
        selected_op_cache.append(op_cache[winner_idx])  # Assuming op_cache is available and aligned with op_population
    return selected_op_genes, selected_op_cache

#@njit(cache=True)
def create_OP_gene(METALEVEL_COUNT, NUMBER_OF_META_OPS, PRIOR_LEVEL_OPS):
    combine_OPs = np.random.randint(0, NUMBER_OF_META_OPS , size=(METALEVEL_COUNT, NUMBER_OF_META_OPS,PRIOR_LEVEL_OPS )).astype(np.float64)   
    cached_metalevels = _generate_cached_metalevels(combine_OPs)
    #0 to 1 issue with metalevel and base ops where meta_level number passed to cached_metalevels should be 1 less than the actual metalevel in the gene 
    # as teh 
    return combine_OPs, cached_metalevels

#@njit(cache=True)
def create_OP_population(METALEVEL_COUNT,NUMBER_OF_META_OPS , PRIOR_LEVEL_OPS):
    combine_OPs = np.empty((METALEVEL_COUNT, NUMBER_OF_META_OPS, PRIOR_LEVEL_OPS), dtype=object)

    for i in range(METALEVEL_COUNT):
        for j in range(NUMBER_OF_META_OPS):
            for k in range(PRIOR_LEVEL_OPS):
                combine_OPs[i,j,k] = create_OP_gene(METALEVEL_COUNT, NUMBER_OF_META_OPS, PRIOR_LEVEL_OPS)

    return combine_OPs

def get_ops_from_meta_level(meta_level, ops_to_resolve ,combine_OPs,target_level=0):
    #METALEVEL_COUNT, NUMBER_OF_OPS, PRIOR_LEVEL_OPS = combine_OPs.shape
    #FIXME does this ignore the first level?
    meta_level = meta_level-1

    if meta_level < target_level:
        return ops_to_resolve
    
    level_OPs_genesis = combine_OPs[meta_level, int(0)]

    for op_idx in ops_to_resolve[1:]: 
        if op_idx == -1: #PAD_OP_VALUE
            continue
        
        level_OPs = combine_OPs[meta_level, int(op_idx)]
        
        level_OPs_genesis = np.hstack((level_OPs_genesis, level_OPs))


        
    return get_ops_from_meta_level(meta_level,level_OPs_genesis, combine_OPs, target_level )
    


#@njit(cache=True)
def _generate_cached_metalevels(combine_OPs, level=0, OP_PAD_VALUE = -1 ):
    METALEVEL_COUNT, NUMBER_OF_OPS, PRIOR_LEVEL_OPS = combine_OPs.shape
    MAX_OPS_COUNT =  PRIOR_LEVEL_OPS**METALEVEL_COUNT
    cached_metalevels = np.zeros(shape=(METALEVEL_COUNT, NUMBER_OF_OPS, MAX_OPS_COUNT))

    for metalevel_idx in range(METALEVEL_COUNT-1, -1, -1):
        for op_number in range(NUMBER_OF_OPS):
            metalevel_OPs = combine_OPs[metalevel_idx, op_number]
            level_1_ops = get_ops_from_meta_level(metalevel_idx, metalevel_OPs, combine_OPs )

            if len(level_1_ops) < MAX_OPS_COUNT:
                ops_to_pad = MAX_OPS_COUNT - len(level_1_ops)
                ops_to_pad = np.ones(shape=(ops_to_pad))*OP_PAD_VALUE
                level_1_ops = np.hstack((level_1_ops, ops_to_pad))

            cached_metalevels[metalevel_idx, op_number] = level_1_ops
            
    return cached_metalevels
