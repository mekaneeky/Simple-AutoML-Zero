def resolve_hierarchical_gene(gene, level_data, ops_dict, max_args):
    if len(gene) == 0:
        return level_data
    else:
        new_level_data = []
        for op_gene in gene[0]:
            op_code = int(op_gene[0])
            op_func = resolve_op(op_code)
            op_args = [level_data[int(arg_location)] for arg_location in op_gene[1:max_args+1]]
            output_location = int(op_gene[max_args+1])
            result = op_func(*op_args, *op_gene[max_args+2:])
            new_level_data.append(result)
        level_data = np.vstack((level_data, np.array(new_level_data)))
        return resolve_hierarchical_gene(gene[1:], level_data, ops_dict, max_args)




def generate_random_hierarchical_gene(num_levels: int, max_ops_per_level: int, max_args: int, 
x_shape: int, y_shape: int, constants_low: int, constants_high: int) -> list:
    gene = []
    for level in range(num_levels):
        level_gene = []
        for op in range(max_ops_per_level):
            op_gene = generate_random_hierarchical_gene(max_ops_per_level, max_args, x_shape, y_shape, constants_low, constants_high)
            level_gene.append(op_gene)
        gene.append(np.array(level_gene))
    return gene