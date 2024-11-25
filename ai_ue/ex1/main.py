import random as rd

import numpy as np

# create random state
start_state = list(range(9))
rd.shuffle(start_state)
np.matrix(
    start_state,
)
# check if state is solveable
# inversion of row
# inversion of column
# unsolveable if [sum of inversions / 2] != 0


# heuristic 1

# time complexity
# space complexity


# heuristic 2

# time complexity
# space complexity


# compare heuristics
print(start_state)
