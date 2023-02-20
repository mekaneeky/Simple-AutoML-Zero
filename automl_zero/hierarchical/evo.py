from automl_zero.evo import generate_random_op_gene



def generate_random_hierarchical_gene(NUMBER_OF_OPS = None, 
    MAX_ARG = 2, X_SHAPE = None,y_shape= None ,OP_DEPTH = None,
    CONSTANTS_LOW = -50, CONSTANTS_HIGH= 50,CONSTANTS_MAX =2,
    initialization = "random", 
    meta_levels = None,
    op_population = None):
    
    if initialization == "random":
        #metalevel limiter?
        #bypass metalevel op?
        OP_gene = np.random.randint(0, NUMBER_OF_OPS , size=(OP_DEPTH,meta_levels,1)).astype(np.float64)
        temp_mem = initialize_memory_limited(X_SHAPE, y_shape) # to have correct arg_list_sizes
        arg_locations = np.random.randint(0, len(temp_mem),size=(OP_DEPTH, MAX_ARG)).astype(np.float64)
        output_locations = np.random.randint(0, len(temp_mem) ,size=(OP_DEPTH,1)).astype(np.float64)
        constants = np.random.uniform(CONSTANTS_LOW,CONSTANTS_HIGH, size=(OP_DEPTH,CONSTANTS_MAX))#gaussian or uniform ??
        #FIXME non int constants

    elif initialization == "zeros":
        OP_gene = np.zeros(shape=(OP_DEPTH,meta_levels,1)).astype(np.float64)
        temp_mem = initialize_memory_limited(X_SHAPE, y_shape) # to have correct arg_list_sizes
        arg_locations = np.zeros(shape=(OP_DEPTH, MAX_ARG)).astype(np.float64)
        output_locations = np.zeros(shape=(OP_DEPTH,1)).astype(np.float64)
        constants = np.zeros(shape=(OP_DEPTH,CONSTANTS_MAX))#gaussian or uniform ??        
        #FIXME non int constants
    
    #OP_gene = np.array(OP_gene, dtype=np.float64)
    #arg_locations = np.array(arg_locations, dtype=np.float64)
    #output_locations = np.array(output_locations, dtype=np.float64)

    return np.hstack((OP_gene, arg_locations, output_locations, constants))



def generate_random_hierarchical_gene(num_levels: int, max_ops_per_level: int, max_args: int, 
x_shape: int, y_shape: int, constants_low: int, constants_high: int) -> list:
    gene = []
    for level in range(num_levels):
        level_gene = []
        for op in range(max_ops_per_level):
            op_gene = generate_random_gene(op_number, max_args, x_shape, y_shape, constants_low, constants_high)
            level_gene.append(op_gene)
        gene.append(np.array(level_gene))
    return gene


def base_resolve_op(op_code):
    if op_number == 0:
        continue
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

###
# A dense layer is the combination of the following ops activation(matmul(A,B) + C)
# which would be represented as op3(op2(op1(A,B ), C))
# Can our representation devolve into a linked list of ops instead? 
# Also can we specify whether the output goes to memory or to another op in sequence? or is it the same under evo search ?
###

def hierarchical_resolve_genome(gene, memory_arr, max_args, population_list):
        
        #(OP_gene, arg_locations, output_locations, constants)

        #OP_gene.shape (OP_depth, num_metalevels)

        #For the first meta-level we apply the OP_gene to the input data
        # The next metalevel we combine the OP_genes of 2 population members
        # and so on for m metalevels
        # We could add a combination basic OP as well 

        if gene
        
        
        return hierarchical_resolve_genome() 

## TODO this can be vectorized
@njit(cache=True)
def hierarchical_run_evolution(X, y, iters = 100000,
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
                accuracy = False \
                ):
    """
    There are 2 modes of referencing 
    """

    #start_time = time.time()
    for i in range(iters):
        
        if i%10000 == 0:
            best_fitness_idx = get_best_fitness(fitness_list)
            print("BEST FITNESS at iteration ")
            print(i)
            print("is ")
            print(fitness_list[best_fitness_idx])
            #print(fitness_list[best_fitness_idx])
            #wandb.config.best_fitness = fitness_list[best_fitness_idx]
            # if fitness_list[best_fitness_idx] == 0:
            #     print("ITER Escaped at: ")
            #     print(i)
            #     return population_list,fitness_list, i
        

        ## Tournament selection
        contestant_indicies = choice(np.arange(len(fitness_list)), size=N, replace=False)
        tournament_winner_idx = run_tournament(fitness_list, contestant_indicies)        
        new_metagene = mutate_winner(population_list[tournament_winner_idx], len(memory_ref_dict))
        ## FIXME How can this cause issues? Leak zeros inappropriatly?
        
        ## Output is zero 
        memory_ref_dict[2][:] = np.zeros(shape=memory_ref_dict[2].shape )
        
        if setup_function:
            resolve_genome(
                                gene = new_metagene["gene_setup"], \
                                memory_ref_dict = memory_ref_dict,
                                resolve_depth = SETUP_OP_DEPTH)

        new_fitness = 0
        new_accuracy = 0
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
            new_accuracy += np.sum(np.floor(preds) == y_val)
            #print(preds == y)
            #print(len(X))
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
        if accuracy:
            fitness_list.append(-new_accuracy/len(X))
        else:
            fitness_list.append(new_fitness)
        fitness_list.pop(0)
    #end_time = time.time()
    #seconds_spent = end_time-start_time
    #iters_per_second = iters/seconds_spent
    #print(f"ITERS PER SECOND: {iters_per_second}" )
    return population_list, fitness_list, i


## If there is no further meta-level then 
 def combine_operations(op1, op2, data):
        
    if op1 == 0 or op2 == 0:
        return data
    
    def combined_op(data1, data2):
        # Perform the operations in a specific order, which can be changed as needed
        result1 = pop_dict[op1](data1)
        result2 = pop_dict[op2](data2)
        return np.add(result1, result2)

    return combined_op

def hierarchical_combine_operations(op_dict: dict, pop_dict: dict, op_idx,
                                    data_types: typing.Tuple[np.dtype, np.dtype],
                                    data_shapes: typing.Tuple[typing.Tuple[int, int], typing.Tuple[int, int]]):
     

    # Generate a new set of operations based on the previous level
    new_op_dict = {}
    for i in range(op_idx):
        for j in range(i, op_idx):
            combined_op = combine_operations(op_dict[i], op_dict[j], data_shapes[0], data_shapes[1])
            new_op_dict[len(new_op_dict)] = combined_op
      
    return new_op_dict