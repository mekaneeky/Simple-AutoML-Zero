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



def hierarchical_combine_operations(op_dict, pop_dict, op_idx,
                                    data_types ,
                                    data_shapes):
     

    # Generate a new set of operations based on the previous level
    new_op_dict = {}
    for i in range(op_idx):
        for j in range(i, op_idx):
            combined_op = combine_operations(op_dict[i], op_dict[j], data_shapes[0], data_shapes[1])
            new_op_dict[len(new_op_dict)] = combined_op
      
    return new_op_dict

def _mini_resolve(op_gene,meta_level, data1, data2):

    if meta_level == 0:
        for gene in op_gene:
            op_to_apply = base_resolve_op(op_gene)
            result = op_to_apply(data1,data2)
            data1 = result
        return result
    
    ### recurse or loop?
def _full_resolve(op_gene, data1, data2):
    raise NotImplementedError

#this should run from resolve or just before it
## Since this is decoupled from the content of the OPs
## It could be used to generate the lower level gene for population members as well?
## Could we do some sort of swarming, collaboration or do I need to sleep? 
def decode_metalevel(current_metalevel,current_OPs, combine_OPs):#current_metalevel comes from OP_metalevel[current_depth]
    
    if current_metalevel == 0:
        return current_OPs
    
    new_OPs = np.array([])

    for OP_depth in current_OPs:
        if len(new_OPs) == 0:
            new_OPs = combine_OPs[current_metalevel,OP_depth] #of size PRIOR_LEVEL_OPS
        else:
            new_decoded_ops = combine_OPs[current_metalevel,OP_depth] #of size PRIOR_LEVEL_OPS
            new_OPs = np.hstack([new_OPs,new_decoded_ops] )

    return decode_metalevel(current_metalevel-1,new_OPs, combine_OPs)

