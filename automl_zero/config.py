import numpy as np
BETA = 0.05
X_arr = np.array([[10,10],
                      
    ]).reshape(-1,2)
X_SHAPE = X_arr.shape
y_true = X_arr // 2
y_shape = y_true.shape
MAXIMUM_ARGUMENT_LENGTH = 2
INPUT_SIZE = 1
#MEMORY_SIZE = 5 + INPUT_SIZE
#MATRIX_SIZE = (5,1) #FIXME doesn't allow for values not == X_val
#MEMORY_SIZE = (5,1)
#MATRIX_SIZE = MEMORY_SIZE
#MEMORY_LOWER = -5
#MEMORY_UPPER = 5
CONSTANTS_LOW = -100
CONSTANTS_HIGH = 100


INPUT_ADDR = 0
OUTPUT_ADDR = 1
INPUT_CONST_A = 2
INPUT_CONST_B = 3
INPUT_CONST_C = 4

OUTPUT_VECTOR_POS = 5
OUTPUT_MATRIX_POS_1 = 6
OUTPUT_MATRIX_POS_2 = 7
#OUTPUT_SHAPE = ()

N_ARGS = 2
OP_DEPTH = 5

MIN_VAL = -1
MIN_FITNESS = 999999

setup_function = True
learn_function = False
random_initialization= True
