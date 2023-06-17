import numpy as np
from numba import njit
from numba.typed import List, Dict
from numba.types import Tuple

#@njit(cache=True)
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
def _generate_cached_metalevels(combine_OPs, level=0, \
                                begin_meta_level_idx = None, begin_op_idx = None,
                                OP_PAD_VALUE = -1 ):
    METALEVEL_COUNT, NUMBER_OF_OPS, PRIOR_LEVEL_OPS = combine_OPs.shape
    MAX_OPS_COUNT =  PRIOR_LEVEL_OPS**METALEVEL_COUNT
    cached_metalevels = np.zeros(shape=(METALEVEL_COUNT, NUMBER_OF_OPS, MAX_OPS_COUNT))#

    for metalevel_idx in range(METALEVEL_COUNT-1, -1, -1):
        all_metalevel_OPs = combine_OPs[metalevel_idx]
        for op_number in range(NUMBER_OF_OPS):
            metalevel_OPs = all_metalevel_OPs[op_number]
            level_1_ops = get_ops_from_meta_level(metalevel_idx, metalevel_OPs, combine_OPs )

            if len(level_1_ops) < MAX_OPS_COUNT:
                ops_to_pad = MAX_OPS_COUNT - len(level_1_ops)
                ops_to_pad = np.ones(shape=(ops_to_pad))*OP_PAD_VALUE
                level_1_ops = np.hstack((level_1_ops, ops_to_pad))

            cached_metalevels[metalevel_idx, op_number] = level_1_ops
            
    return cached_metalevels


#FIXME inefficient for all 3 to share the same METALEVEL_COUNT, NUMBER_OF_OPS, PRIOR_LEVEL_OPS
# all 3 meta_op genes should have separate sizes
#@njit(cache=True)
def create_OP_population(METALEVEL_COUNT, NUMBER_OF_OPS, PRIOR_LEVEL_OPS,
                        SETUP_BASE_OPS_COUNT = None, # 0 
                        PRED_BASE_OPS_COUNT = None,  # 1 
                        LEARN_BASE_OPS_COUNT = None, # 2
                        POPULATION_SIZE = 10):
    _COUNT_OF_GENES = 3

    combine_OPs = np.random.randint(0, NUMBER_OF_OPS , size=(POPULATION_SIZE,_COUNT_OF_GENES ,METALEVEL_COUNT, NUMBER_OF_OPS,PRIOR_LEVEL_OPS )).astype(np.float64)   
    combine_OPs[:,0,:,:,:] = np.random.randint(0, SETUP_BASE_OPS_COUNT-1 , size=(POPULATION_SIZE, 1, NUMBER_OF_OPS,PRIOR_LEVEL_OPS )).astype(np.float64)
    combine_OPs[:,1,:,:,:] = np.random.randint(0, PRED_BASE_OPS_COUNT-1 , size=(POPULATION_SIZE, 1, NUMBER_OF_OPS,PRIOR_LEVEL_OPS )).astype(np.float64)
    combine_OPs[:,2,:,:,:] = np.random.randint(0, LEARN_BASE_OPS_COUNT-1 , size=(POPULATION_SIZE, 1, NUMBER_OF_OPS,PRIOR_LEVEL_OPS )).astype(np.float64)

    cached_metalevels = []
    for pop_idx in range(POPULATION_SIZE):
        op_cached_ops_dict = Dict()

        setup_cached_metalevel_arr = _generate_cached_metalevels(combine_OPs[pop_idx,0])
        pred_cached_metalevel_arr = _generate_cached_metalevels(combine_OPs[pop_idx,1])
        learn_cached_metalevel_arr = _generate_cached_metalevels(combine_OPs[pop_idx,2])
        op_cached_ops_dict["gene_setup"] = setup_cached_metalevel_arr
        op_cached_ops_dict["gene_pred"] = pred_cached_metalevel_arr
        op_cached_ops_dict["gene_learn"] = learn_cached_metalevel_arr
        cached_metalevels.append(op_cached_ops_dict)
    
    #cached_metalevels = np.hstack(cached_metalevels) 
    return combine_OPs, cached_metalevels

#@njit(cache=True)
def create_OP_gene(METALEVEL_COUNT, NUMBER_OF_META_OPS, PRIOR_LEVEL_OPS):
    combine_OPs = np.random.randint(0, NUMBER_OF_META_OPS , size=(METALEVEL_COUNT, NUMBER_OF_META_OPS,PRIOR_LEVEL_OPS )).astype(np.float64)   
    cached_metalevels = _generate_cached_metalevels(combine_OPs)
    #0 to 1 issue with metalevel and base ops where meta_level number passed to cached_metalevels should be 1 less than the actual metalevel in the gene 
    # as teh 
    return combine_OPs, cached_metalevels
