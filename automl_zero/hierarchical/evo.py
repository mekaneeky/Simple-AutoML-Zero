from automl_zero.hierarchical.config import *
from automl_zero.memory import initialize_memory_limited
from automl_zero.ops import *
from automl_zero.hierarchical.mutators import mutate_winner, mutate_combination_winner
from automl_zero.hierarchical.op_combination import create_OP_population, get_ops_from_meta_level, _generate_cached_metalevels
from numpy.random import choice
from numba import njit
from numba.typed import List, Dict

#@njit(cache=True)
def generate_random_hierarchical_gene(NUMBER_OF_BASE_OPS = SETUP_OP_NUMBER, 
    MAX_ARG = 2, X_SHAPE = None,y_shape= None ,OP_DEPTH = OP_DEPTH,
    CONSTANTS_LOW = -50, CONSTANTS_HIGH= 50,CONSTANTS_MAX =2,
    initialization = "random", 
    METALEVEL_COUNT = METALEVEL_COUNT,
    PRIOR_LEVEL_OPS = PRIOR_LEVEL_OPS,
    NUMBER_OF_META_OPS = NUMBER_OF_META_OPS,
    built_in_ops = False,
    scalars = None, vectors = None, matricies=None
):

    """
    Generate a random hierarchical gene for use in Hierarchical AutoML Zero.

    Args:
        NUMBER_OF_OPS (int): The number of operations in the operation set.
        MAX_ARG (int): The maximum number of arguments each operation can take.
        X_SHAPE (Tuple[int, int]): The shape of the input data.
        y_shape (Tuple[int, int]): The shape of the target data.
        OP_DEPTH (int): The depth of the hierarchical gene.
        CONSTANTS_LOW (float): The lower bound for the random constants in the gene.
        CONSTANTS_HIGH (float): The upper bound for the random constants in the gene.
        CONSTANTS_MAX (int): The maximum number of constants each operation can use.
        initialization (str): The initialization method for the gene. Can be "random" or "zeros".
        METALEVEL_COUNT (int): The number of meta-levels in the gene.

    Returns:
        np.ndarray: A random hierarchical gene with shape (OP_DEPTH, NUMBER_OF_OPS*2 + MAX_ARG + 1 + CONSTANTS_MAX).

    Raises:
        ValueError: If initialization is not "random" or "zeros".
    """

    if initialization == "random":

        base_OP_idx = np.random.randint(0, NUMBER_OF_META_OPS-1 , size=(OP_DEPTH,1)).astype(np.int8) #(OP_DEPTH, OP_IDX, METALEVEL)
        OP_metalevel = np.random.randint(0, METALEVEL_COUNT-1 , size=(OP_DEPTH,1)).astype(np.int8) #(OP_DEPTH, OP_IDX, METALEVEL)
        for op_idx in range(OP_DEPTH):
            if OP_metalevel[op_idx] == 0:
                base_OP_idx[op_idx] = np.random.randint(0, NUMBER_OF_BASE_OPS-1 , size=(1)).astype(np.int8) #probably introduces leakage. Should just ignore and let evolution
                #weed them out
        #TODO make combine OPs not fixed to number of ops 

        temp_mem = initialize_memory_limited(X_SHAPE, y_shape, scalars=scalars,vectors=vectors,matricies=matricies) # to have correct arg_list_sizes
        arg_locations = np.random.randint(0, len(temp_mem)-1,size=(OP_DEPTH, MAX_ARG)).astype(np.float64)
        output_locations = np.random.randint(0, len(temp_mem)-1 ,size=(OP_DEPTH,1)).astype(np.float64)
        constants = np.random.uniform(CONSTANTS_LOW,CONSTANTS_HIGH, size=(OP_DEPTH,CONSTANTS_MAX))#gaussian or uniform ??
        #FIXME non int constants


        #if built_in_ops:
        #    combine_OPs = np.random.randint(0, NUMBER_OF_BASE_OPS-1 , size=(METALEVEL_COUNT, NUMBER_OF_META_OPS,PRIOR_LEVEL_OPS )).astype(np.float64)

    #TODO add gradual metalevel expansions
    elif initialization == "zeros":
        base_OP_idx = np.zeros(shape=(OP_DEPTH,1)).astype(np.int8) #(OP_DEPTH, OP_IDX, METALEVEL)
        OP_metalevel = np.zeros(shape=(OP_DEPTH,1)).astype(np.int8) #(OP_DEPTH, OP_IDX, METALEVEL)

        temp_mem = initialize_memory_limited(X_SHAPE, y_shape,scalars=scalars,vectors=vectors,matricies=matricies) # to have correct arg_list_sizes
        arg_locations = np.zeros(shape=(OP_DEPTH, MAX_ARG)).astype(np.float64)
        output_locations = np.zeros(shape=(OP_DEPTH,1)).astype(np.float64)
        constants = np.zeros(shape=(OP_DEPTH,CONSTANTS_MAX))#gaussian or uniform ??        
        #FIXME non int constants

        #TODO Hashing caching thing for ops broken down by the decoder function
        #if built_in_ops:
        #    combine_OPs = np.zeros(shape=(METALEVEL_COUNT, NUMBER_OF_META_OPS,PRIOR_LEVEL_OPS )).astype(np.float64)


    #cached_metalevels[OP_metalevel, metalevel_OP_value]
    #if built_in_ops:
    #    cached_metalevels = _generate_cached_metalevels(combine_OPs)

    #if built_in_ops:
    #    return combine_OPs, cached_metalevels, np.hstack((base_OP_idx, OP_metalevel, arg_locations, output_locations, constants))
    #else:    
    return np.hstack((base_OP_idx, OP_metalevel, arg_locations, output_locations, constants))#cached_metalevels, combine_OPs, 



## TODO How can I make this more efficient and have the combinations here maybe?
#@njit(cache=True)
def base_resolve_op(op_number):
    if op_number == 0:
        result = add_scalar#FIXME
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

#@njit(cache=True)
def run_tournament(fitness_list, contestant_indicies):
    min_fitness = 9999999999.0
    min_idx = -1
    for contestant_idx in contestant_indicies:
        if fitness_list[contestant_idx] <= min_fitness:
            min_fitness = fitness_list[contestant_idx]
            min_idx = contestant_idx
    
    winner_idx = min_idx
    
    return winner_idx

#@njit(cache=True)
def initialize_gene_population(X = None, y = None, \
                          memory_ref_dict = None, \
                          fitness_func=None, \
                          #setup_OP_dict = None, SETUP_OP_DEPTH = None , \
                          #pred_OP_dict = None, PRED_OP_DEPTH = None, \
                          #learn_OP_dict = None, LEARN_OP_DEPTH = None, \
                          #setup_function = False,
                          #learn_function = False,
                          #initialization= "zeros",
                          #accuracy = False,
                          #METALEVEL_COUNT = None,
                          cached_metalevels = None,
                          ):
    

    """
    Initialize a population of genes for evolutionary optimization.

    Args:
        X (np.ndarray, optional): Input data matrix. Defaults to None.
        y (np.ndarray, optional): Target data matrix. Defaults to None.
        population_count (int, optional): Number of genes to generate in the population. Defaults to POPULATION_COUNT.
        memory_ref_dict (dict, optional): Dictionary that maps memory indices to memory buffers. Defaults to None.
        fitness_func (function, optional): Fitness function used to evaluate the performance of each gene. Defaults to None.
        max_OP_depth (int, optional): Maximum depth of the operation hierarchy in each gene. Defaults to None.
        setup_OP_dict (dict, optional): Dictionary of operations used to construct the setup function in each gene. Defaults to None.
        SETUP_OP_DEPTH (int, optional): Depth of the setup operation hierarchy in each gene. Defaults to None.
        pred_OP_dict (dict, optional): Dictionary of operations used to construct the prediction function in each gene. Defaults to None.
        PRED_OP_DEPTH (int, optional): Depth of the prediction operation hierarchy in each gene. Defaults to None.
        learn_OP_dict (dict, optional): Dictionary of operations used to construct the learning function in each gene. Defaults to None.
        LEARN_OP_DEPTH (int, optional): Depth of the learning operation hierarchy in each gene. Defaults to None.
        setup_function (bool, optional): Whether to include the setup function in each gene. Defaults to False.
        learn_function (bool, optional): Whether to include the learning function in each gene. Defaults to False.
        initialization (str, optional): Initialization method used to generate the genes. Defaults to "zeros".
        accuracy (bool, optional): Whether to calculate accuracy as a fitness metric. Defaults to False.

    Returns:
        tuple: Tuple containing the list of generated genes and their corresponding fitness values.
    """
    ## Generate lists of populations with 2 options 
    ## 1 carrying genes and metagenes
    ## 2 each metagene is an element in itself with its own memory and genes

    ## Each metagene then constructs its op_dict or equivalent from the operations of the prior gene 
    ## this needs to be a separate function because this can be repeated during run_evolution/training
    
    genes_list = List()
    fitness_list = List()
    for i in range(POPULATION_COUNT):


        if i%10 == 0:
            print(i)

        gene_dict = Dict()

        if setup_function:
            # Only one memory for both? 
            gene_setup = generate_random_hierarchical_gene(NUMBER_OF_BASE_OPS = SETUP_OP_NUMBER, 
                                                           NUMBER_OF_META_OPS= SETUP_OP_NUMBER,
                                              X_SHAPE = X[0].shape,
                                              y_shape = y[0].shape,
                                              OP_DEPTH = max_OP_depth,
                                              initialization = "random" if random_initialization else "zeros",
                                              scalars = scalars,vectors = vectors,matricies = matricies,
                                              METALEVEL_COUNT=METALEVEL_COUNT)
        else:
            gene_setup = None

        gene_pred = generate_random_hierarchical_gene(NUMBER_OF_BASE_OPS = PRED_OP_NUMBER,
                                                      NUMBER_OF_META_OPS= PRED_OP_NUMBER,
                                        X_SHAPE = X[0].shape, 
                                        y_shape = y[0].shape,
                                        OP_DEPTH = max_OP_depth,
                                        initialization = "random" if random_initialization else "zeros",
                                        scalars = scalars,vectors = vectors,matricies = matricies,
                                        METALEVEL_COUNT=METALEVEL_COUNT)

        

        if learn_function:
            gene_learn = generate_random_hierarchical_gene(NUMBER_OF_BASE_OPS = LEARN_OP_NUMBER, 
                                                           NUMBER_OF_META_OPS= LEARN_OP_NUMBER,
                                                X_SHAPE = X[0].shape,
                                                y_shape = y[0].shape,
                                                OP_DEPTH = max_OP_depth,
                                                initialization = "random" if random_initialization else "zeros",
                                                scalars = scalars,vectors = vectors,matricies = matricies,
                                                METALEVEL_COUNT=METALEVEL_COUNT)

        
        else:
            gene_learn = None

        import pdb;pdb.set_trace()
        fitness = 0.0
        accuracy = 0.0
        for x_idx in range(len(X)):            
            X_val = X[x_idx:x_idx+1]
            y_val = y[x_idx:x_idx+1]
            #remove previous y used in learn to avoid leakage if repeat value/single example
            memory_ref_dict[2][:] = np.zeros(shape=memory_ref_dict[2].shape )
            #FIXME move out of loop
            #setup function
            if setup_function:
                #try:
                hierarchical_resolve_genome(
                                    gene = gene_setup, \
                                    cached_metalevels = cached_metalevels,#["gene_setup"], \
                                    resolve_depth = SETUP_OP_DEPTH,
                                    memory_ref_dict = memory_ref_dict,
                                    MAX_ARG=MAX_ARG)
                #except:
                #    pass #log or some shit

            #predict function
            memory_ref_dict[0][:] = X_val
            try:
                preds = hierarchical_resolve_genome(
                                    gene = gene_pred, \
                                    cached_metalevels = cached_metalevels["gene_pred"], \
                                    resolve_depth = PRED_OP_DEPTH,
                                    memory_ref_dict= memory_ref_dict,
                                    MAX_ARG=MAX_ARG  )# we can supply different OP_dicts to shift meta-levels
                #print(memory_ref_dict)
                
                #fitness is for a single pred here not all? 
                # we do a sum ? 
                fitness += fitness_func(preds, y_val)
                accuracy += np.sum(np.floor(preds) == y_val)
            except:
                accuracy = 0
                fitness = float(9999999999.000)

            #print("Learn for real")

            if learn_function:
                # Add real y to allow for learning 
                memory_ref_dict[2][:] = y_val
                try:
                    hierarchical_resolve_genome(
                            gene = gene_learn, \
                            cached_metalevels = cached_metalevels["gene_learn"], \
                            resolve_depth = LEARN_OP_DEPTH,
                            memory_ref_dict= memory_ref_dict,
                            MAX_ARG=MAX_ARG  )# we can supply different OP_dicts to shift meta-levels
                except:
                    pass

        if accuracy:
            fitness_list.append(-accuracy/len(X))
        else:
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





###
# A dense layer is the combination of the following ops activation(matmul(A,B) + C)
# which would be represented as op3(op2(op1(A,B ), C))
# Can our representation devolve into a linked list of ops instead? 
# Also can we specify whether the output goes to memory or to another op in sequence? or is it the same under evo search ?
###
#@njit(cache=True)
def hierarchical_resolve_genome(
                            gene,
                            cached_metalevels,
                            memory_ref_dict, #separate memory per level or mixed
                            resolve_depth = None,
                            constants_flag = False,
                            OP_depth = None,
                            MAX_ARG = None):
        
        #(OP_gene, arg_locations, output_locations, constants)

        #OP_gene.shape (OP_depth, num_metalevels)

        #For the first meta-level we apply the OP_gene to the input data
        # The next metalevel we combine the OP_genes of 2 population members
        # and so on for m metalevels
        # We could add a combination basic OP as well 

        #genome_OPs, args_locations, output_locations, constants = gene[:,0], gene[:,1:3], gene[:,3], gene[:,4:6]

        #OP_gene = np.zeros(shape=(OP_DEPTH,METALEVEL_COUNT)).astype(np.float64)
        #combine_OPs = np.random.randint(0, NUMBER_OF_OPS , size=(OP_DEPTH, METALEVEL_COUNT)).astype(np.float64)
        #temp_mem = initialize_memory_limited(X_SHAPE, y_shape) # to have correct arg_list_sizes
        #arg_locations = np.zeros(shape=(OP_DEPTH, MAX_ARG)).astype(np.float64)
        #output_locations = np.zeros(shape=(OP_DEPTH,1)).astype(np.float64)
        #constants = np.zeros(shape=(OP_DEPTH,CONSTANTS_MAX))#gaussian or uniform ??        

        # combine_OPs
    #np.hstack((base_OP_idx, OP_metalevel, arg_locations, output_locations, constants))#cached_metalevels, combine_OPs, 
    genome_OPs, metalevel_gene, args_locations, output_locations, constants = gene[:,0:1], gene[:,1:2], gene[:,2:2+MAX_ARG], gene[:,2+MAX_ARG:3+MAX_ARG],gene[:,3+MAX_ARG:]
    #print(metalevel_gene)
    for gene_idx in range(len(genome_OPs)):
        OP_value = genome_OPs[gene_idx].astype(np.int8)
        metalevel = metalevel_gene[gene_idx].astype(np.int8)


        if metalevel == 0:
            first_level_OPs = [OP_value]
        else:
            try:
                first_level_OPs = cached_metalevels[metalevel, OP_value]
            except: 
                import pdb;pdb.set_trace()

        #TODO add the input and output injection bit? Add OP for memory assign?
        try:
            op_args_0 = memory_ref_dict[int(args_locations[gene_idx][0])]
            op_args_1 = memory_ref_dict[int(args_locations[gene_idx][1])]
            first_OP_flag = True
        except:
            import pdb;pdb.set_trace()

        for OP_idx in first_level_OPs:
            
            try:
                OP_to_execute = base_resolve_op(OP_idx)
            except: #-1
                break
            

            if first_OP_flag:
                temp_result = OP_to_execute(op_args_0, op_args_1)
                first_OP_flag = False
            else:
                temp_result = OP_to_execute(temp_result, op_args_1) # I don't like the way this looks
                # How can this allow for different inputs for different layers? 
                # or combined outputs such as DARTS? 

        output_idx = int(output_locations[gene_idx])

        try:
            if temp_result is not None: 
                memory_ref_dict[output_idx][:] = temp_result
            else:
                memory_ref_dict[output_idx][:] = np.zeros(shape= memory_ref_dict[output_idx].shape)
        except:
            #memory_ref_dict[output_idx][:] = np.resize(result, output_arr.shape)
            memory_ref_dict[output_idx][:] = np.zeros(shape= memory_ref_dict[output_idx].shape)

    return memory_ref_dict[1]




## So the combine OPs is of shape 
## Combine OPs is the main OP gene dict then the resulting OPs are generated 
## through applying the hierarchical replresentation that we can have a linear 
## linked list, however if we want to move to a DAG? We don't need to
## we can let the entire program be the dag? 

#@njit(cache=True)
def get_best_fitness(fitness_list):
    min_fitness = 9999999999
    for gene_idx in range(len(fitness_list)):
        if min_fitness >= fitness_list[gene_idx]:
            min_fitness = fitness_list[gene_idx]
            min_idx = gene_idx
    return min_idx

## TODO this can be vectorized
#@njit(cache=True)
def hierarchical_run_gene_evolution(X, y, iters = 100000,
                evaluate_steps = 10000, \
                fitness_func = None, \
                population_list = None, \
                fitness_list = None, \
                setup_function = True, \
                setup_OP_dict = setup_OPs, \
                SETUP_OP_DEPTH = None, \
                pred_OP_dict = pred_OPs, \
                PRED_OP_DEPTH = None, \
                learn_function = True, \
                learn_OP_dict = learn_OPs, \
                LEARN_OP_DEPTH = None, \
                N=10, \
                memory_ref_dict = None, \
                accuracy = False, \
                cached_metalevels = None,
                MAX_ARG = 2,
                #evolve_OPs = True,
                #evolve_genes = True
                ):
    """
    There are 2 modes of referencing 
    """
    #start_time = time.time()
    for i in range(iters):
        
        if i%10 == 0:
            best_fitness_idx = get_best_fitness(fitness_list)
            print("BEST FITNESS at iteration ")
            print(i)
            print("is ")
            print(fitness_list[best_fitness_idx])
            
        ## Tournament selection
        contestant_indicies = choice(np.arange(len(fitness_list)), size=N, replace=False)
        tournament_winner_idx = run_tournament(fitness_list, contestant_indicies)   
        import pdb;pdb.set_trace()     
        new_metagene = mutate_winner(population_list[tournament_winner_idx], len(memory_ref_dict))
        ## FIXME How can this cause issues? Leak zeros inappropriatly?
        
        ## Output is zero 
        memory_ref_dict[2][:] = np.zeros(shape=memory_ref_dict[2].shape )
        
        if setup_function:
            hierarchical_resolve_genome(
                                gene = new_metagene["gene_setup"], \
                                cached_metalevels = cached_metalevels, \
                                memory_ref_dict = memory_ref_dict,
                                resolve_depth = SETUP_OP_DEPTH,
                                MAX_ARG = MAX_ARG)

        new_fitness = 0
        new_accuracy = 0
        #TODO add in f(x)? profile
        #TODO profile this zip vs range(len())
        for X_val, y_val in zip(X,y):        
            
            #remove previous y used in learn to avoid leakage if repeat value/single example
            memory_ref_dict[2][:] = np.zeros(shape=memory_ref_dict[2].shape )

            #predict function
            memory_ref_dict[0][:] = X_val
            preds = hierarchical_resolve_genome(
                                gene = new_metagene["gene_pred"], \
                                cached_metalevels = cached_metalevels, \
                                memory_ref_dict= memory_ref_dict,
                                resolve_depth=PRED_OP_DEPTH,
                                MAX_ARG = MAX_ARG  )# we can supply different OP_dicts to shift meta-levels
            
            #fitness is for a single pred here not all? 
            # we do a sum ? 
            new_fitness += fitness_func(preds, y_val)
            new_accuracy += np.sum(np.floor(preds) == y_val)
            if learn_function:
                # Add real y to allow for learning 
                memory_ref_dict[2][:] = y_val

                hierarchical_resolve_genome(
                        gene = new_metagene["gene_learn"], \
                        cached_metalevels = cached_metalevels, \
                        memory_ref_dict= memory_ref_dict,
                        resolve_depth=LEARN_OP_DEPTH,
                        MAX_ARG = MAX_ARG)# we can supply different OP_dicts to shift meta-levels

        ## Append to population list
        population_list.append( new_metagene )
        population_list.pop(0)
        if accuracy:
            fitness_list.append(-new_accuracy/len(X))
        else:
            fitness_list.append(new_fitness)
        fitness_list.pop(0)

    return population_list, fitness_list, i



#Write a function run_hierarchical_op_evolution that takes in a population list and fitness list and a lsit of combine ops to evolve
## TODO this can be vectorized
#@njit(cache=True)
def hierarchical_run_op_evolution(X, y, iters = 100000,
                evaluate_steps = 100, \
                fitness_func = None, \
                population_list = None, \
                op_fitness_list = None, \
                setup_function = True, \
                setup_OP_dict = setup_OPs, \
                SETUP_OP_DEPTH = None, \
                pred_OP_dict = pred_OPs, \
                PRED_OP_DEPTH = None, \
                learn_function = True, \
                learn_OP_dict = learn_OPs, \
                LEARN_OP_DEPTH = None, \
                N=10, \
                memory_ref_dict = None, \
                accuracy = False, \
                combine_OPs = None,
                cached_metalevels = None,
                ):
    
    base_ops_dict = Dict()
    base_ops_dict["gene_setup"] = 12#Needs to be hardcoded for now
    base_ops_dict["gene_pred"] = 12#Needs to be hardcoded for now
    base_ops_dict["gene_learn"] = 12#Needs to be hardcoded for now
    
    if op_fitness_list is None:
        op_fitness_list = [0]*len(combine_OPs)
    
    for i in range(iters):
    
        if i%10000 == 0:
            best_fitness_idx = get_best_fitness(op_fitness_list)
            print("BEST FITNESS at iteration ")
            print(i)
            print("is ")
            print(op_fitness_list[best_fitness_idx])
            
        ## Tournament selection
        contestant_indicies = choice(np.arange(len(op_fitness_list)), size=N, replace=False)
        tournament_winner_idx = run_tournament(op_fitness_list, contestant_indicies)        
        new_metagene = mutate_winner(population_list[tournament_winner_idx], len(memory_ref_dict))
        ## FIXME How can this cause issues? Leak zeros inappropriatly?
        
        ## Output is zero 
        memory_ref_dict[2][:] = np.zeros(shape=memory_ref_dict[2].shape )
        

        for combined_OPs_idx in range(len(combine_OPs)):
            #combine_OP_member = combine_OPs[combined_OPs_idx]
            cached_metalevels_member = cached_metalevels[combined_OPs_idx]

            for metagene in population_list:
                for X_val, y_val in zip(X,y):   

                    if setup_function:
                        hierarchical_resolve_genome(
                                            gene = new_metagene["gene_setup"], \
                                            cached_metalevels = cached_metalevels_member, \
                                            memory_ref_dict = memory_ref_dict,
                                            resolve_depth = SETUP_OP_DEPTH)

                    new_fitness = 0
                    new_accuracy = 0
                    #TODO add in f(x)? profile
                    #TODO profile this zip vs range(len())
                        
                                
                    #remove previous y used in learn to avoid leakage if repeat value/single example
                    memory_ref_dict[2][:] = np.zeros(shape=memory_ref_dict[2].shape )

                    #predict function
                    memory_ref_dict[0][:] = X_val
                    preds = hierarchical_resolve_genome(
                                        gene = new_metagene["gene_pred"], \
                                        cached_metalevels = cached_metalevels_member, \
                                        memory_ref_dict= memory_ref_dict,
                                        resolve_depth=PRED_OP_DEPTH  )# we can supply different OP_dicts to shift meta-levels
                    
                    #fitness is for a single pred here not all? 
                    # we do a sum ? 
                    new_fitness += fitness_func(preds, y_val)
                    new_accuracy += np.sum(np.floor(preds) == y_val)
                    if learn_function:
                        # Add real y to allow for learning 
                        memory_ref_dict[2][:] = y_val

                        hierarchical_resolve_genome(
                                gene = new_metagene["gene_learn"], \
                                cached_metalevels = cached_metalevels_member, \
                                memory_ref_dict= memory_ref_dict,
                                resolve_depth=LEARN_OP_DEPTH  )# we can supply different OP_dicts to shift meta-levels

                ## Append to population list
                population_list.append( new_metagene )
                population_list.pop(0)
                if accuracy:
                    op_fitness_list.append(-new_accuracy/len(X))
                else:
                    op_fitness_list.append(new_fitness)
                op_fitness_list.pop(0)

    return population_list, op_fitness_list, i
