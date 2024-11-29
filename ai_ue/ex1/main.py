import random as rd
import time

import numpy as np

# create random state
start_state = list(range(9))
rd.shuffle(start_state)
matrix = np.array(start_state).reshape(-1, 3)

# goal state
goal_matrix = np.array(list(range(9))).reshape(-1, 3)


# check if state is solveable
# inversion counter
def get_inv_count(arr: np.ndarray) -> int:
    inv_count = 0
    for i in range(arr.size):
        for j in range(i + 1, arr.size):
            if arr[j] != 0 and arr[i] != 0 and arr[i] > arr[j]:
                inv_count += 1
    return inv_count


# check if puzzle is solveable via [sum of inversions mod 29 = 0
def puzzle_solvable(matrix: np.matrix) -> bool:
    return get_inv_count(matrix.flatten()) % 2 == 0


# heuristic 1: Hamming distance heuristic
# compare each number in matrix with goal matrix and if they differ add one to the hamming distance
# we have to exclude the blank space to get the correct distance
def get_hamming_distance(matrix: np.matrix, goal_matrix: np.matrix) -> int:
    return sum(
        num != goal_num and num != 0
        for num, goal_num in zip(matrix.flatten(), goal_matrix.flatten(), strict=True)
    )


# time complexity
# space complexity


# heuristic 2: Manhattan distance
def get_manhattan_distance(matrix: np.matrix, goal_matrix: np.matrix) -> int:
    distance = 0
    for row in matrix:
        for num in row:
            goal_row, goal_column = np.asarray(np.where(goal_matrix == num)).flatten()
            m_row, m_column = np.asarray(np.where(matrix == num)).flatten()
            distance += abs(m_row - goal_row) + abs(m_column - goal_column)
    return distance


# time complexity
# space complexity


# generate successors: generate next possible moves
def get_next_possible_moves(matrix: np.matrix):
    next_moves = []
    m_row, m_column = np.asarray(np.where(matrix == 0)).flatten()
    # if row index on top border only add bottom as possible move
    # if row index on bottom border only add top as possible move
    # if row index in center add top and bottom as possible move
    # same with column and left/right


# compare heuristics

print(f"start state: \n{matrix}")

solveable = puzzle_solvable(matrix)
print(f"solveable?: {puzzle_solvable(matrix)}")
if solveable:
    print(f"hamming distance: {get_hamming_distance(matrix,goal_matrix)}")
    print(f"manhattan distance: {get_manhattan_distance(matrix,goal_matrix)}")
