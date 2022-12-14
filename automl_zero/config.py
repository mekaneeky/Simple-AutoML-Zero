import numpy as np
BETA = 0.05
X_arr = np.array([[10,22, 5000, 300, 999],
                      [1012,242, 50200, 3002, 999],
                      [1013,232, 50400, 3001, 999],
                      [101,212, 52000, 3300, 4999],
                      [1,2, 500, 30, 99],
                      [120,222, 50200, 3200, 9929],
    ]).reshape(-1,5,1)
X_SHAPE = X_arr.shape
y_true = X_arr / 2
y_shape = y_true.shape
MAXIMUM_ARGUMENT_LENGTH = 2
INPUT_SIZE = 1
#MEMORY_SIZE = 5 + INPUT_SIZE
#MATRIX_SIZE = (5,1) #FIXME doesn't allow for values not == X_val
MEMORY_SIZE = (5,1)
MATRIX_SIZE = MEMORY_SIZE
MEMORY_LOWER = -5
MEMORY_UPPER = 5
CONSTANTS_LOW = -100
CONSTANTS_HIGH = 100

POPULATION_COUNT = 1000
TOURNAMENT_COUNT = 10
ITERS = 5000000

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