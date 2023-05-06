# GPT3

def _generate_cached_metalevels(combine_OPs, OP_depth, level=0,
                                begin_meta_level_idx=None, begin_op_idx=None,
                                modified_meta_level=None,
                                OP_PAD_VALUE=-1):
    # METALEVEL_COUNT, NUMBER_OF_OPS,PRIOR_LEVEL_OPS
    METALEVEL_COUNT, NUMBER_OF_OPS, PRIOR_LEVEL_OPS = combine_OPs.shape
    MAX_OPS_COUNT = NUMBER_OF_OPS * METALEVEL_COUNT
    cached_metalevels = np.zeros(
        shape=(NUMBER_OF_OPS * METALEVEL_COUNT, NUMBER_OF_OPS * METALEVEL_COUNT, MAX_OPS_COUNT))

    if modified_meta_level is None:
        modified_meta_level = np.zeros(METALEVEL_COUNT, dtype=bool)

    if begin_meta_level_idx is None:
        begin_meta_level_idx = METALEVEL_COUNT

    if begin_op_idx is None:
        begin_op_idx = 0

    OP_padding_value = -1  # np.pad numba support
    for metalevel_idx in range(begin_meta_level_idx, 0, -1):
        all_metalevel_OPs = combine_OPs[metalevel_idx]
        for op_number in range(begin_op_idx, NUMBER_OF_OPS):
            metalevel_OPs = all_metalevel_OPs[op_number]
            if not modified_meta_level[metalevel_idx] and all(
                    not modified_meta_level[x] for x in metalevel_OPs):
                continue

            level_1_ops = get_ops_from_meta_level(metalevel_idx, metalevel_OPs, combine_OPs)

            if len(level_1_ops) < MAX_OPS_COUNT:
                ops_to_pad = MAX_OPS_COUNT - len(level_1_ops)
                ops_to_pad = np.ones(shape=(ops_to_pad)) * OP_PAD_VALUE
                level_1_ops = np.hstack([level_1_ops, ops_to_pad])

            cached_metalevels[metalevel_idx, op_number] = level_1_ops

        # mark modified_meta_level as True for all metalevels that depend on the current one
        modified_meta_level[np.where(all_metalevel_OPs[:, :] == op_number)] = True

    return cached_metalevels