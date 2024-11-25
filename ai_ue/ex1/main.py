import random as rd

import numpy as np

# create random state
start_state = list(range(9))
rd.shuffle(start_state)
matrix = np.array(start_state).reshape(-1, 3)


print(start_state)
print(matrix)


# check if state is solveable
# inversion counter
def get_inv_count(arr: np.ndarray):
    inv_count = 0
    for i in range(arr.size):
        for j in range(i + 1, arr.size):
            if arr[j] != 0 and arr[i] != 0 and arr[i] > arr[j]:
                inv_count += 1
    return inv_count


# check if puzzle is solveable via [sum of inversions mod 29 = 0
def puzzle_solvable(matrix: np.matrix):
    return get_inv_count(matrix.flatten()) % 2 == 0


# heuristic 1

# time complexity
# space complexity


# heuristic 2

# time complexity
# space complexity


# compare heuristics
print(f"solveable?: {puzzle_solvable(matrix)}")
