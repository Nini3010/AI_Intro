import heapq
import random as rd
import time
from math import sqrt

import numpy as np

# create random state
puzzle_size = 9

start_state = list(range(puzzle_size))
rd.shuffle(start_state)
matrix = np.array(start_state).reshape(-1, int(sqrt(puzzle_size)))


# goal state
GOAL_MATRIX = np.array(list(range(9))).reshape(-1, int(sqrt(puzzle_size)))


# check if state is solveable
# inversion counter
def get_inv_count(arr: np.ndarray) -> int:
    inv_count = 0
    for i in range(arr.size):
        for j in range(i + 1, arr.size):
            if arr[j] != 0 and arr[i] != 0 and arr[i] > arr[j]:
                inv_count += 1
    return inv_count


# check if puzzle is solveable via [sum of inversions] mod 29 = 0
def puzzle_solvable(matrix: np.matrix) -> bool:
    return get_inv_count(matrix.flatten()) % 2 == 0


# heuristic 1: Hamming distance heuristic
# compare each number in matrix with goal matrix and if they differ add one to the hamming distance
# we have to exclude the blank space to get the correct distance
def get_hamming_distance(matrix: np.matrix) -> int:
    return sum(
        num != goal_num and num != 0
        for num, goal_num in zip(matrix.flatten(), GOAL_MATRIX.flatten(), strict=True)
    )


# time complexity
# space complexity


# heuristic 2: Manhattan distance
def get_manhattan_distance(matrix: np.matrix) -> int:
    distance = 0
    for row in matrix:
        for num in row:
            goal_row, goal_column = np.asarray(np.where(GOAL_MATRIX == num)).flatten()
            m_row, m_column = np.asarray(np.where(matrix == num)).flatten()
            distance += abs(m_row - goal_row) + abs(m_column - goal_column)
    return distance


# time complexity
# space complexity


# generate successors: generate next possible moves
def get_next_possible_moves(matrix: np.matrix) -> list:
    next_empty_space_coords = []
    next_moves = []
    empty_row, empty_column = np.asarray(np.where(matrix == 0)).flatten()
    # if row index not on top border add top as possible move
    if empty_row > 0:
        next_empty_space_coords.append((empty_row - 1, empty_column))
    # if row index not on bottom border add bottom as possible move
    if empty_row < matrix.shape[0] - 1:
        next_empty_space_coords.append((empty_row + 1, empty_column))
    # if column index not on left border add left as possible move
    if empty_column > 0:
        next_empty_space_coords.append((empty_row, empty_column - 1))
    # if column index not on right border add right as possible move
    if empty_column < matrix.shape[1] - 1:
        next_empty_space_coords.append((empty_row, empty_column + 1))

    # create copy of current stage matrix and generate successor by making possible move in copy
    for row, column in next_empty_space_coords:
        next_matrix = matrix.copy()
        next_matrix[empty_row][empty_column], next_matrix[row][column] = (
            next_matrix[row][column],
            next_matrix[empty_row][empty_column],
        )
        next_moves.append(next_matrix)
    return next_moves


# missing: costs + current step
def calc_costs(next_moves: list[np.matrix], heuristic: callable) -> int:
    h_costs: list[int] = []
    for move in next_moves:
        h_costs.append(heuristic(move))
    return h_costs


# compare heuristics

print(f"start state: \n{matrix}")
solveable = puzzle_solvable(matrix)
print(f"solveable?: {puzzle_solvable(matrix)}")
if solveable:
    print(f"heuristic?: \n1: Hamming\n2: Manhattan\n9: Benchmark\n(input number)")
    heuristic_num = int(input())
    if heuristic_num == 1:
        heuristic = get_hamming_distance
    elif heuristic_num == 2:
        heuristic = get_manhattan_distance
    if heuristic is not None:
        unsolved = True
        queue: heapq = []
        current_step = 1
        while unsolved:
            moves = get_next_possible_moves(matrix)
            costs = calc_costs(moves, heuristic)
            priorities = [x + current_step for x in costs]
            # moves = [move for _, move in sorted(zip(priorities, moves, strict=True))]
            # costs = [cost for _, cost in sorted(zip(priorities, costs, strict=True))]
            # priorities = sorted(priorities)

            queue = heapq.heapify(
                [
                    (prio, cost, move)
                    for prio, cost, move in zip(priorities, costs, moves, strict=True)
                ]
            )
            print(f"heap: {queue}")
            unsolved = False
            # for prio,move in zip(priorities,moves):

            # print(f"hamming distance: {get_hamming_distance(matrix,GOAL_MATRIX)}")
            # print(f"manhattan distance: {get_manhattan_distance(matrix,GOAL_MATRIX)}")
            # print(f"successors stage 1: {get_next_possible_moves(matrix)}")
